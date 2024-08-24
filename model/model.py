#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from io import StringIO # Python 3.
from datasets import load_dataset,Dataset,DatasetDict,concatenate_datasets

from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import pandas as pd
import json
import pickle
from transformers import AdamW, get_scheduler
from datasets import load_metric

from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from saveAndLoad import *

from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split

class SelfAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SelfAttentionClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.dense_layer = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # Compute queries, keys, and values
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        
        # Compute attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        
        # Apply attention to values
        weighted = torch.bmm(attention, values)
        
        # Skip connection and LayerNorm (first layer)
        weighted = self.layer_norm1(weighted + x)
        
        # Pass through the dense layer and ReLU activation
        new_vec = self.dense_layer(weighted)
        new_vec = self.relu(new_vec)
        
        # Skip connection and LayerNorm (second layer)
        new_vec = self.layer_norm2(new_vec + weighted)
        
        # Apply dropout
        sequence_outputs = self.dropout(new_vec)
        
        # Classifier applied on the first token's output
        logits = self.classifier(sequence_outputs[:, 0, :])
        
        return logits

# Example usage:
# model = SelfAttentionNetwork(input_dim=768, hidden_dim=768, num_classes=10)
# x = torch.rand(32, 10, 768)  # Batch size of 32, sequence length of 10, embedding size of 768
# logits = model(x)


#REPLACE ED_ENCODER WITH EMBEDDING LOOKUP
# ED_ENCODER RETURNS LAST_HIDDEN_STATE
# class ED_encoder(nn.Module):
#     """
#     A task-specific custom transformer model for predicting ED Disposition. 
#     This model loads a pre-trained transformer model and adds a new dropout 
#     and linear layer at the end for fine-tuning and prediction on specific tasks.
#     """
#     def __init__(self, checkpoint, num_labels, freeze=True):
#         """
#         Args:
#             checkpoint (str): The name of the pre-trained model or path to the model weights.
#             num_labels (int): The number of output labels in the final classification layer.
#         """
#         super(ED_encoder, self).__init__()
#         self.num_labels = num_labels # number of labels for classifier
        
#         # checkpoint is the model name 
#         self.model = model = AutoModel.from_pretrained(checkpoint, config = AutoConfig.from_pretrained(checkpoint, 
#                                                                                                        output_attention = True, 
#                                                                                                        output_hidden_state = True ) )
#         if freeze:
#             for parameter in self.model.parameters():
#                 parameter.requires_grad = False
        
#     def forward(self, input_ids = None, attention_mask=None, labels = None ):
#         """
#         Forward pass for the model.
        
#         Args:
#             input_ids (torch.Tensor, optional): Tensor of input IDs. Defaults to None.
#             attention_mask (torch.Tensor, optional): Tensor for attention masks. Defaults to None.
#             labels (torch.Tensor, optional): Tensor for labels. Defaults to None.
            
#         Returns:
#             TokenClassifierOutput: A named tuple with the following fields:
#             - loss (torch.FloatTensor of shape (1,), optional, returned when label_ids is provided) – Classification loss.
#             - logits (torch.FloatTensor of shape (batch_size, num_labels)) – Classification scores before SoftMax.
#             - hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) – Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
#             - attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) – Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
#         """
#         # calls on the Automodel to deploy correct model - in our case distilled-bert-uncased
#         outputs = self.model(input_ids = input_ids, attention_mask = attention_mask  )
        
#         # retrieves the last hidden state
#         last_hidden_state = outputs[0]
        
#         return last_hidden_state # The embedding

# x is embeddings from ED_classifier
class Classifier(nn.Module):
    def __init__(self, input_dim, n_labels, dropout=.3):
        super(Classifier, self).__init__()
        self.input_dim = input_dim 
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))  # Learnable CLS token
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        self.num_labels = n_labels # number of labels for classifier
        self.dropout = nn.Dropout(dropout) # to prevent overfitting
        self.classifier = nn.Linear(input_dim, n_labels) # FC Layer - takes in a 768 token vector and is a Linear classifier with n labels
        self.dense_layer = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()  # ReLU non-linearity
        self.loss_func = nn.CrossEntropyLoss() # Change this if it becomes more than binary classification

    def forward(self, x, labels=None):
        # typical self attention workflow
        # print(x.shape)
        is_pad = x == float('-inf')
        pad_rows = is_pad.all(dim=2)
        print(pad_rows.shape)
        print(pad_rows)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand CLS token for each sequence in the batch
        x = torch.cat((cls_tokens, x), dim=1)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        #potentially add dropout here
        weighted = torch.bmm(attention, values)
        new_vec = self.dense_layer(weighted)
        new_vec = self.relu(new_vec)  # Apply ReLU activation
        sequence_outputs = self.dropout(new_vec)
        classifier_input = sequence_outputs[:, 0, :].view(-1, self.input_dim)
        logits = self.classifier(classifier_input)

        # print(f'queries = Linear({self.input_dim},{self.input_dim})', queries.shape, '\n')        
        # print(f'keys = Linear({self.input_dim},{self.input_dim})', keys.shape, '\n')        
        # print(f'values = Linear({self.input_dim},{self.input_dim})', values.shape, '\n')        
        # print('scores = bmm(queries, keys.transpose(1,2)) / (self.input_dim ** 0.5)', scores.shape, '\n')
        # print('attention = Softmax(dim=2)(scores)', attention.shape, '\n')
        # print('weighted = bmm(attention, values)', weighted.shape, '\n')
        # print('sequence_outputs = Linear(768,768), relu, dropout (weighted)', sequence_outputs.shape, '\n')
        # print('sequence_outputs[:, 0, :].view(-1, self.input_dim)', classifier_input.shape,'\n')
        # print('logits = Linear(768, 2)(sequence_outputs[:, 0, :].view(-1,768))', logits.shape,'\n')
        
        loss = None
        if labels is not None:
            loss = self.loss_func(logits.view(-1, self.num_labels), labels.view(-1))
            
            # TokenClassifierOutput - returns predicted label
            output = TokenClassifierOutput(loss=loss, logits=logits)
            print(output)
            return output
        
        else:
            return logits


# In[ ]:


canonical_mut_embeddings_esm2 = np.load('../aa/canonical_mut_embeddings_esm2.npy')

labeled_data = pd.read_csv('../data_processing/cancer_type_detailed_data.csv')
data = labeled_data['0'].values

label_encoder = LabelEncoder()
string_labels = labeled_data['1']
labels = label_encoder.fit_transform(string_labels)
labels = torch.tensor(labels,dtype=torch.long)

class Dataset_MutationList(Dataset):
    def __init__(self, data, labels, embeddings):
        load_str = lambda x: list(map(int,x.split(',')))
        self.data = [load_str(i) for i in data]
        self.labels = labels
        self.embeddings = embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idxs = self.data[idx]
        emb = torch.stack([torch.tensor(self.embeddings[i],dtype=torch.float32) for i in idxs])
        return emb, self.labels[idx]
    
def custom_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = pad_sequence(data, batch_first=True, padding_value=float('-inf'))
    labels = torch.stack(labels)
    return data, labels

# Create dataset
dataset = Dataset_MutationList(data, labels, canonical_mut_embeddings_esm2)

# Create DataLoader
# dataloader = DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=custom_collate)


# In[ ]:


test_size = .2
random_state = 42
batch_size = 1
indices = list(range(len(dataset)))

train_indices, test_indices = train_test_split(
    indices, 
    test_size=test_size, 
    random_state=random_state
)

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
    
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)


# In[ ]:


import torch.optim as optim

model = Classifier(input_dim=640, n_labels=804)
# model = SelfAttentionClassifier(input_dim=640, hidden_dim=640, num_classes=804)

num_epochs = 20
learning_rate = 0.001

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        # assert False
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')


# In[ ]:





# In[ ]:




