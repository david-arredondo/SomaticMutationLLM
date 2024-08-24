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

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(keepdim=True, dim=-1) * (x.size(-1) ** -0.5)
        return self.scale * (x / (norm + self.eps))
    
class MLP(nn.Module):

    def __init__(self, n_embd, dropout, bias):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

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
        self.classifier = nn.Linear(input_dim, n_labels) # FC Layer
        self.dense_layer = nn.Linear(input_dim, input_dim)
        self.rms_norm = RMSNorm(input_dim)  # RMS normalization
        self.relu = nn.ReLU()  # ReLU non-linearity
        self.loss_func = nn.CrossEntropyLoss() # Change this if it becomes more than binary classification
        self.mlp = MLP(input_dim, dropout, False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        is_pad = x == float('-inf')
        pad_rows = is_pad.all(dim=2)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand CLS token for each sequence in the batch
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.rms_norm(x)

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)

        # Apply dense layer
        new_vec = self.dense_layer(weighted)

        x = x + new_vec

        # Apply RMS normalization
        x = self.rms_norm(x)

        # Apply ReLU activation
        x = x + self.mlp(x)

        sequence_outputs = new_vec
        classifier_input = sequence_outputs[:, 0, :].view(-1, self.input_dim)
        logits = self.classifier(classifier_input)

        return logits


# In[56]:

device = 'cuda:1'
canonical_mut_embeddings_esm2 = np.load('../aa/canonical_mut_embeddings_esm2.npy')
data_dir = '../labeled_data/'
labeled_data = os.listdir(data_dir)
for ni,i in enumerate(labeled_data):print(ni,i)
data = labeled_data[4]
print('\n',data)
data_df = pd.read_csv(data_dir+data)
data = data_df['idxs'].values
nlabels = len(data_df['int_label'].unique())
labels = torch.tensor(data_df['int_label'].values,dtype=torch.long)
device = 'cuda:1'

class Dataset_MutationList(Dataset):
    def __init__(self, data, labels, embeddings,device):
        load_str = lambda x: list(map(int,x.split(',')))
        self.data = [load_str(i) for i in data]
        self.labels = labels
        self.embeddings = embeddings
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idxs = self.data[idx]
        emb = torch.stack([torch.tensor(self.embeddings[i],dtype=torch.float32) for i in idxs])
        return emb.to(self.device), self.labels[idx].to(self.device)
    
def custom_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = pad_sequence(data, batch_first=True, padding_value=float('-inf'))
    labels = torch.stack(labels)
    return data, labels

# Create dataset
dataset = Dataset_MutationList(data, labels, canonical_mut_embeddings_esm2, device)

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
from tqdm import tqdm
model = Classifier(input_dim=640, n_labels=nlabels)
model.to(device)
# model = SelfAttentionClassifier(input_dim=640, hidden_dim=640, num_classes=804)

num_epochs = 50
learning_rate = 0.001

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    with tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
        for batch_idx, (data, target) in pbar:
            optimizer.zero_grad()
            output = model(data)
            # assert False
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Epoch':f'{epoch+1}/{num_epochs}, Loss: {loss.item():.4f}'})
            if batch_idx % 20000 == 0:
                print(f'Loss: {loss.item():.4f}')

        # Evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(test_loader):
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')