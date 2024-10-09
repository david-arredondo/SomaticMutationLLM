import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
from transformers import AdamW, get_scheduler
from datasets import load_metric

from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from saveAndLoad import *

from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split

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
        self.config = config

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(-2, -1)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        y = torch.bmm(attention, values)
        return self.dense_layer(y)

class Block(nn.Module):
    def __init__(self, config, norm_fn = nn.LayerNorm):
        super().__init__()
        self.norm1 = config.norm_fn(config.input_dim)  # RMS normalization
        self.norm2 = config.norm_fn(config.input_dim)
        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

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
        self.classifier = nn.Linear(config.input_dim + config.binary_dim, config.n_labels) # FC Layer
        self.loss_func = nn.CrossEntropyLoss() # Change this if it becomes more than binary classification
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, binary):
        # is_pad = x == float('-inf')
        # pad_rows = is_pad.all(dim=2)
        # x = self.emb_transform(x)

        if self.pooling == 'cls':
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand CLS token for each sequence in the batch
            x = torch.cat((cls_tokens, x), dim=1)
        
        for block in self.blocks:
            x = block(x)

        if self.pooling == 'cls':
            classifier_input = x[:, 0, :].view(-1, self.input_dim)
        elif self.pooling == 'mean':
            classifier_input = x.mean(dim=1)
        elif self.pooling == 'max':
            classifier_input, _ = x.max(dim=1)

        classifier_input = torch.cat((classifier_input, binary), dim=1)

        logits = self.classifier(classifier_input)
        return logits