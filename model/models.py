import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
from transformers import AdamW, get_scheduler
from datasets import load_metric
import torch.optim as optim
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from saveAndLoad import *

from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split


## SINGLE HEAD SELF-ATTENTION
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, bias = None):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(keepdim=True, dim=-1) * (x.size(-1) ** -0.5)
        return self.scale * (x / (norm + self.eps))
    
class MLP(nn.Module):

    def __init__(self, config, use_dropout=True):
        super().__init__()
        self.c_fc    = nn.Linear(config.input_dim, 4 * config.input_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.input_dim, config.input_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        if self.use_dropout: x = self.dropout(x)
        return x
    
# class Attention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.input_dim = config.input_dim
#         self.query = nn.Linear(config.input_dim, config.input_dim)
#         self.key = nn.Linear(config.input_dim, config.input_dim)
#         self.value = nn.Linear(config.input_dim, config.input_dim)
#         self.softmax = nn.Softmax(dim=2)
#         self.dense_layer = nn.Linear(config.input_dim, config.input_dim)
#         self.dropout = nn.Dropout(config.dropout)
#         self.add_pos = config.position_embedding
#         self.pos_embs = nn.Embedding(config.max_len, config.input_dim)
#         self.config = config

#     def forward(self, x, positions=None):
#         if self.add_pos: #position embeddings for top N
#             pos = torch.arange(0, x.shape[-2], dtype=torch.long, device=x.device)
#             pos_emb = self.pos_embs(pos)
#             x = x + pos_emb
#         if positions is not None: #hard coded positions for assays
#             pos_emb = self.pos_embs(positions)
#             x = x + pos_emb
#         queries = self.query(x)
#         keys = self.key(x)
#         values = self.value(x)
#         scores = torch.bmm(queries, keys.transpose(-2, -1)) / (self.input_dim ** 0.5)
#         attention = self.softmax(scores)
#         y = torch.bmm(attention, values)
#         return self.dense_layer(y)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.input_dim
        self.query = nn.Linear(config.input_dim, config.input_dim)
        self.key = nn.Linear(config.input_dim, config.input_dim)
        self.value = nn.Linear(config.input_dim, config.input_dim)
        self.softmax = nn.Softmax(dim=2)
        self.dense_layer = nn.Linear(config.input_dim, config.input_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.add_pos = config.position_embedding
        self.pos_embs = nn.Embedding(config.max_len, config.input_dim)
        self.config = config

    def forward(self, x, positions=None):
        if self.add_pos: #position embeddings for top N
            pos = torch.arange(0, x.shape[-2], dtype=torch.long, device=x.device)
            pos_emb = self.pos_embs(pos)
            x = x + pos_emb
        if positions is not None: #hard coded positions for assays
            pos_emb = self.pos_embs(positions)
            x = x + pos_emb
        queries = self.query(x)
        keys = self.key(x)
        values = torch.nan_to_num(self.value(x),nan=0)
        scores = torch.bmm(queries, keys.transpose(-2, -1)) / (self.input_dim ** 0.5)
        scores = torch.nan_to_num(scores, nan=-torch.inf)
        attention = self.softmax(scores)
        y = torch.bmm(attention, values)
        d = self.dense_layer(y)
        return d

class Block(nn.Module):
    def __init__(self, config, norm_fn = nn.LayerNorm):
        super().__init__()
        self.norm1 = config.norm_fn(config.input_dim)  # RMS normalization
        self.norm2 = config.norm_fn(config.input_dim)
        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x, positions=None):
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.mlp(self.norm2(x))
        return x

# class Classifier(nn.Module):
#     def __init__(self, config):
#         super(Classifier, self).__init__()

#         self.blocks = nn.ModuleList([Block(config, norm_fn = config.norm_fn) for _ in range(config.n_layer)])

#         self.input_dim = config.input_dim 
#         self.pooling = config.pooling
#         assert self.pooling in ['cls', 'mean', 'max'], 'pooling should be either cls, mean, or max'
#         self.cls_token = nn.Parameter(torch.randn(1, 1, config.input_dim))  # Learnable CLS token

#         # self.emb_transform = nn.Linear(config.input_dim, config.input_dim)  
#         # self.emb_transform = MLP(config,use_dropout=False)  

#         self.num_labels = config.n_labels # number of labels for classifier
#         self.classifier = nn.Linear(config.input_dim, config.n_labels) # FC Layer
#         self.loss_func = nn.CrossEntropyLoss() # Change this if it becomes more than binary classification
        
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)

#     def forward(self, x, positions = None):
#         # is_pad = x == 0
#         # pad_rows = is_pad.all(dim=2)
#         # x = self.emb_transform(x)

#         if self.pooling == 'cls':
#             batch_size = x.size(0)
#             cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand CLS token for each sequence in the batch
#             x = torch.cat((cls_tokens, x), dim=1)
        
#         for block in self.blocks:
#             x = block(x,positions)

#         if self.pooling == 'cls':
#             classifier_input = x[:, 0, :].view(-1, self.input_dim)
#         elif self.pooling == 'mean':
#             classifier_input = x.mean(dim=1)
#         elif self.pooling == 'max':
#             classifier_input, _ = x.max(dim=1)

#         logits = self.classifier(classifier_input)
#         return logits

class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()

        self.blocks = nn.ModuleList([Block(config, norm_fn = config.norm_fn) for _ in range(config.n_layer)])

        self.input_dim = config.input_dim 
        self.pooling = config.pooling
        assert self.pooling in ['cls', 'mean', 'max'], 'pooling should be either cls, mean, or max'
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.input_dim))  # Learnable CLS token

        # self.emb_transform = nn.Linear(config.input_dim, config.input_dim)  
        # self.emb_transform = MLP(config,use_dropout=False)  

        self.num_labels = config.n_labels # number of labels for classifier
        self.classifier = nn.Linear(config.input_dim, config.n_labels) # FC Layer
        self.loss_func = nn.CrossEntropyLoss() # Change this if it becomes more than binary classification
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, positions = None):
        is_pad = x == 0
        x[is_pad] = -torch.inf

        if self.pooling == 'cls':
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand CLS token for each sequence in the batch
            x = torch.cat((cls_tokens, x), dim=1)
        
        for block in self.blocks:
            x = block(x,positions)

        if self.pooling == 'cls':
            classifier_input = x[:, 0, :].view(-1, self.input_dim)
        elif self.pooling == 'mean':
            classifier_input = torch.nanmean(x,dim=1)
        elif self.pooling == 'max':
            classifier_input, _ = x.max(dim=1)
        logits = self.classifier(classifier_input)
        return logits


## NAIVE BINARY AND TOP 10
class MLPClassifier(nn.Module):
    def __init__(self, config):
        super(MLPClassifier, self).__init__()

        self.input_dim = config.input_dim 
        self.num_labels = config.n_labels # number of labels for classifier
        self.linear1 = nn.Linear(config.input_dim,256)
        self.linear2 = nn.Linear(256,256)
        self.classifier = nn.Linear(256, config.n_labels) # FC Layer
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        logits = self.classifier(x)
        return logits
    
class BigMLP(nn.Module):
    def __init__(self, config):
        super(BigMLP, self).__init__()
        self.input_dim = config.input_dim 
        self.num_labels = config.n_labels # number of labels for classifier
        layers = [
            nn.Linear(self.input_dim,2048),
            nn.BatchNorm1d(2048),
            nn.Linear(2048,1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.Linear(512, config.n_labels) # FC Layer
        ]
        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.model(x)
    
class LRClassifier(nn.Module):
    def __init__(self, config):
        super(LRClassifier, self).__init__()
        self.input_dim = config.input_dim 
        self.num_labels = config.n_labels # number of labels for classifier
        self.classifier = nn.Linear(config.input_dim,config.n_labels)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        logits = self.classifier(x)
        return logits
    
def train(model,num_epochs,train_loader,test_loader, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        with tqdm(enumerate(train_loader), total=len(train_loader),desc='TRAINING') as pbar:
            for batch_idx, (data, target) in pbar:
                optimizer.zero_grad()
                output = model(data)
                # assert False
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'Epoch':f'{epoch+1}/{num_epochs}, Loss: {loss.item():.4f}'})
                # if batch_idx % 20000 == 0:
                #     print('')

            # Evaluation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for data, target in tqdm(test_loader,desc='TESTING'):
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            accuracy = 100 * correct / total
            print(f'Test Accuracy: {accuracy:.2f}%, ({correct} of {total})')

def train_assay(model,num_epochs,train_loader,test_loader, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        with tqdm(enumerate(train_loader), total=len(train_loader),desc='TRAINING') as pbar:
            for batch_idx, (data, positions, target) in pbar:
                optimizer.zero_grad()
                output = model(data, positions)
                # assert False
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'Epoch':f'{epoch+1}/{num_epochs}, Loss: {loss.item():.4f}'})
                # if batch_idx % 20000 == 0:
                #     print('')

            # Evaluation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for data, positions, target in tqdm(test_loader,desc='TESTING'):
                    output = model(data, positions)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            accuracy = 100 * correct / total
            print(f'Test Accuracy: {accuracy:.2f}%, ({correct} of {total})')

## TEST/TRAIN SPLIT

def getTrainTestLoaders(dataset, test_size=.2, random_state=42, batch_size=100, collate=None):
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state
    )

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate)

    return train_loader, test_loader