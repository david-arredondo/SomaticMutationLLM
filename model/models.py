import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
from transformers import AdamW, get_scheduler
# from datasets import load_metric
import torch.optim as optim
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from saveAndLoad import *

from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit

from lifelines.utils import concordance_index

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

    def forward(self, x, positions=None, mask = None):
        if self.add_pos: #position embeddings for top N
            pos = torch.arange(0, x.shape[-2], dtype=torch.long, device=x.device)
            pos_emb = self.pos_embs(pos)
            x = x + pos_emb
        if positions is not None: #hard coded positions for assays
            pos_emb = self.pos_embs(positions)
            x = x + pos_emb
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(-2, -1)) / (self.input_dim ** 0.5)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(scores.size())
            scores = scores.masked_fill(mask == 0, -torch.inf)
        attention = self.softmax(scores)
        y = torch.bmm(attention, values)
        d = self.dense_layer(y)
        return d
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.input_dim
        self.num_heads = config.num_heads
        self.head_dim = self.input_dim // self.num_heads
        
        # Ensure input_dim is divisible by num_heads
        assert self.input_dim % self.num_heads == 0, "input_dim must be divisible by num_heads"
        
        # Layers for projecting inputs to queries, keys, and values
        self.query = nn.Linear(self.input_dim, self.input_dim)
        self.key = nn.Linear(self.input_dim, self.input_dim)
        self.value = nn.Linear(self.input_dim, self.input_dim)
        
        # Final linear layer to combine the heads' outputs
        self.dense_layer = nn.Linear(self.input_dim, self.input_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout)
        
        self.add_pos = config.position_embedding
        self.pos_embs = nn.Embedding(config.max_len, self.input_dim)
        self.config = config

    def forward(self, x, positions=None, mask=None):
        batch_size = x.size(0)
        
        if self.add_pos:  # Add position embeddings
            pos = torch.arange(0, x.shape[-2], dtype=torch.long, device=x.device)
            pos_emb = self.pos_embs(pos)
            x = x + pos_emb
            
        if positions is not None:  # Hard-coded positions for assays
            pos_emb = self.pos_embs(positions)
            x = x + pos_emb
            
        # Compute queries, keys, and values
        queries = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Adjust mask shape for attention heads
            scores = scores.masked_fill(mask == 0, -torch.inf)
        
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # Apply attention to the values
        y = torch.matmul(attention, values)
        
        # Concatenate heads and apply the final linear layer
        y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.input_dim)
        d = self.dense_layer(y)
        
        return d


class Block(nn.Module):
    def __init__(self, config, norm_fn = nn.LayerNorm):
        super().__init__()
        self.norm1 = config.norm_fn(config.input_dim)  # RMS normalization
        self.norm2 = config.norm_fn(config.input_dim)
        if config.num_heads == 1:
            self.attn = Attention(config)
        else:
            self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, positions=None, mask = None):
        n1 = self.norm1(x)
        x = x + self.attn(n1, positions, mask)
        n2 = self.norm2(x)
        x = x + self.mlp(n2)
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
        self.norm = config.norm_fn(config.input_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, positions = None, return_embedding = False):
        if self.pooling == 'cls':
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand CLS token for each sequence in the batch
            x = torch.cat((cls_tokens, x), dim=1)

        mask = (x != 0).any(dim=-1).float()  # Mask out zero vectors
        
        for block in self.blocks:
            x = block(x,positions,mask)

        if self.pooling == 'cls':
            classifier_input = x[:, 0, :].view(-1, self.input_dim)
        elif self.pooling == 'mean':
            mask = mask.unsqueeze(-1).expand_as(x)
            classifier_input = (x * mask).sum(dim=-2) / mask.sum(dim=-2)
        elif self.pooling == 'max':
            mask = mask.unsqueeze(-1).expand_as(x)
            classifier_input, _ = (x * mask).max(dim=-2)

        if self.num_labels==1: classifier_input = self.norm(classifier_input) #survival analysis

        logits = self.classifier(classifier_input)
        if return_embedding:
            return logits, classifier_input
        return logits

class SoMatt_LateFusion(nn.Module):
    def __init__(self, config):
        super(SoMatt_LateFusion, self).__init__()

        self.blocks = nn.ModuleList([Block(config, norm_fn = config.norm_fn) for _ in range(config.n_layer)])

        self.input_dim = config.input_dim 
        self.fusion_dim = config.fusion_dim
        self.pooling = config.pooling
        assert self.pooling in ['cls', 'mean', 'max'], 'pooling should be either cls, mean, or max'
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.input_dim))  # Learnable CLS token

        self.lin1 = nn.Linear(config.fusion_dim, config.fusion_dim)
        self.lin2 = nn.Linear(config.fusion_dim, config.input_dim)
        self.relu = nn.ReLU()

        self.num_labels = config.n_labels # number of labels for classifier
        self.classifier = nn.Linear(config.input_dim * 2, config.n_labels) # FC Layer
        self.norm = config.norm_fn(config.input_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    # x is somatt input, z is late fusion input
    def forward(self, input, return_embedding = False):
        x,positions,z = input
        if self.pooling == 'cls':
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Expand CLS token for each sequence in the batch
            x = torch.cat((cls_tokens, x), dim=1)

        mask = (x != 0).any(dim=-1).float()  # Mask out zero vectors
        
        for block in self.blocks:
            x = block(x,positions,mask)

        if self.pooling == 'cls':
            classifier_input = x[:, 0, :].view(-1, self.input_dim)
        elif self.pooling == 'mean':
            mask = mask.unsqueeze(-1).expand_as(x)
            classifier_input = (x * mask).sum(dim=-2) / mask.sum(dim=-2)
        elif self.pooling == 'max':
            mask = mask.unsqueeze(-1).expand_as(x)
            classifier_input, _ = (x * mask).max(dim=-2)

        z = self.lin1(z)
        z = self.relu(z)
        z = self.lin2(z)
        z = self.relu(z)

        classifier_input = torch.cat((classifier_input,z),dim=1)

        if self.num_labels==1: classifier_input = self.norm(classifier_input) #survival analysis

        logits = self.classifier(classifier_input)
        if return_embedding:
            return logits, classifier_input
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

    def forward(self, x, return_embedding = False):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        logits = self.classifier(x)
        x = self.relu(x)
        if return_embedding:
            return logits, x
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
    
def train(model,num_epochs,train_loader,test_loader, learning_rate=0.001, saveName = None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best = 0,0
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
            if accuracy > best[0]:
                best = accuracy,epoch
                if saveName is not None:
                    torch.save(model.state_dict(), saveName)
                    print(f'Saved {saveName} at epoch {epoch}')
            print(f'Test Accuracy: {accuracy:.2f}%, ({correct} of {total})')
    print(f'Best Accuracy: {best[0]:.2f}% at epoch {best[1]}')

def train_assay(model,num_epochs,train_loader,test_loader, criterion, optimizer, saveName = None):
    best = 0,0
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
            if accuracy > best[0]:
                best = accuracy,epoch
                if saveName is not None:
                    torch.save(model.state_dict(), saveName)
                    print(f'Saved {saveName} at epoch {epoch}')
            print(f'Test Accuracy: {accuracy:.2f}%, ({correct} of {total})')
    print(f'Best Accuracy: {best[0]:.2f}% at epoch {best[1]}')

def negative_log_partial_likelihood(survival, risk, debug=False):
    """Return the negative log-partial likelihood of the prediction
    y_true contains the survival time
    risk is the risk output from the neural network
    censor is the vector of inputs that are censored
    censor data: 1 - dead, 0 - censor
    regularization is the regularization constant (not used currently in model)

    Uses the torch backend to perform calculations

    Sorts the surv_time by sorted reverse time
    https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/deep_surv.py
    """

    # calculate negative log likelihood from estimated risk\
    # print(torch.stack([survival[:, 0], risk]))
    _, idx = torch.sort(survival[:, 0], descending=True)
    censor = survival[idx, 1]
    risk = risk[idx]
    epsilon = 0.00001
    max_value = 10
    alpha = 0.1
    risk = torch.reshape(risk, [-1])  # flatten
    shift = torch.max(risk)
    risk = risk - shift
    # hazard_ratio = torch.exp(risk)

    # cumsum on sorted surv time accounts for concordance
    # log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + epsilon)
    log_risk = torch.logcumsumexp(risk, dim=0)
    log_risk = torch.reshape(log_risk, [-1])
    uncensored_likelihood = risk - log_risk

    # apply censor mask: 1 - dead, 0 - censor
    censored_likelihood = uncensored_likelihood * censor
    num_observed_events = torch.sum(censor)
    if num_observed_events == 0:
        neg_likelihood = torch.tensor(0.0, device=risk.device, requires_grad=True)
    else:
        neg_likelihood = - torch.sum(censored_likelihood) / (num_observed_events)
    return neg_likelihood

def c_index(predicted_risk, survival):
    if survival is None:
        return 0
    # calculate the concordance index
    ci = np.nan  # just to know that concordance index cannot be estimated
    # print(r2python.cbind(np.reshape(predicted_risk, (-1, 1)), survival))

    try:
        na_inx = ~(np.isnan(survival[:, 0]) | np.isnan(survival[:, 1]) | np.isnan(predicted_risk))
        predicted_risk, survival = predicted_risk[na_inx], survival[na_inx]
        if len(predicted_risk) > 0 and sum(survival[:, 1] == 1) > 2:
            survival_time, censor = survival[:, 0], survival[:, 1]
            epsilon = 0.001
            partial_hazard = np.exp(-(predicted_risk + epsilon))
            censor = censor.astype(int)
            ci = concordance_index(survival_time, partial_hazard, censor)

    except:
        ci = np.nan

    return ci

def train_assay_survival(model,num_epochs,train_loader,test_loader, criterion, optimizer, saveName = None):
    best = 0,0
    for epoch in range(num_epochs):
        model.train()
        with tqdm(enumerate(train_loader), total=len(train_loader),desc='TRAINING') as pbar:
            for batch_idx, (data, positions, time, event) in pbar:
                optimizer.zero_grad()
                output = model(data, positions)
                # assert False
                survival = torch.stack([time, event], dim=1)
                loss = criterion(survival, output)
                if torch.isnan(loss):
                    print(survival)
                    print(output)
                    assert False
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'Epoch':f'{epoch+1}/{num_epochs}, Loss: {loss.item():.4f}'})
                # if batch_idx % 20000 == 0:
                #     print('')

            # Evaluation
            model.eval()
            all_times = []
            all_events = []
            all_risks = []

            with torch.no_grad():
                for data, positions, time, event in tqdm(test_loader,desc='TESTING'):
                    output = model(data, positions)
                    risk = output.squeeze()
                    all_risks.extend(risk.cpu().numpy())
                    all_times.extend(time.cpu().numpy())
                    all_events.extend(event.cpu().numpy())
            survival = np.column_stack((all_times, all_events))
            c_index_value = c_index(np.array(all_risks), survival)
            if c_index_value > best[0]:
                best = c_index_value,epoch
                if saveName is not None:
                    torch.save(model.state_dict(), saveName)
                    print(f'Saved {saveName} at epoch {epoch}')
            print(f'C-Index: {c_index_value:.4f}')
    print(f'Best C-Index: {best[0]:.4f} at epoch {best[1]}')

def train_binary_survival(model,num_epochs,train_loader,test_loader, criterion, optimizer, saveName = None):
    best = 0,0
    for epoch in range(num_epochs):
        model.train()
        with tqdm(enumerate(train_loader), total=len(train_loader),desc='TRAINING') as pbar:
            for batch_idx, (data, time, event) in pbar:
                optimizer.zero_grad()
                output = model(data)
                survival = torch.stack([time, event], dim=1)
                loss = criterion(survival, output)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'Epoch':f'{epoch+1}/{num_epochs}, Loss: {loss.item():.4f}'})
                # if batch_idx % 20000 == 0:
                #     print('')

            # Evaluation
            all_times = []
            all_events = []
            all_risks = []
            model.eval()
            with torch.no_grad():
                for data, time, event in tqdm(test_loader,desc='TESTING'):
                    output = model(data)
                    risk = output.squeeze()
                    all_risks.extend(risk.cpu().numpy())
                    all_times.extend(time.cpu().numpy())
                    all_events.extend(event.cpu().numpy())
            survival = np.column_stack((all_times, all_events))
            c_index_value = c_index(np.array(all_risks), survival)
            if c_index_value > best[0]:
                best = c_index_value,epoch
                if saveName is not None:
                    torch.save(model.state_dict(), saveName)
                    print(f'Saved {saveName} at epoch {epoch}')
            print(f'C-Index: {c_index_value:.4f}')
    print(f'Best C-Index: {best[0]:.4f} at epoch {best[1]}')

## TEST/TRAIN SPLIT

def getTrainTestLoaders(dataset, test_size=.2, random_state=42, batch_size=100, collate=None, return_indices = False):
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

    if return_indices:
        return train_loader, test_loader, train_indices, test_indices

    return train_loader, test_loader

def getPatientGroupedLoaders(dataset, data_df, n_folds=None, test_size = None, batch_size=100, collate=None, return_indices=False, random_state=42):
    X = np.arange(len(data_df))
    groups = data_df['patient_id'].values

    if n_folds is not None: f = GroupKFold(n_splits=n_folds)
    elif test_size is not None: f = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    else: assert False, 'Either n_folds or test_size must be provided'
    
    splits = f.split(X, y=None, groups=groups)

    folds = []
    for train_indices, test_indices in splits:
        train_dataset = Subset(dataset, train_indices)
        test_dataset   = Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

        folds.append((train_loader, test_loader))

    return folds

    

########################################
# Attention
########################################

#-----------Cross Attention-------------#

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=config.input_dim, 
                                                    num_heads=config.num_heads, 
                                                    dropout=config.dropout, 
                                                    batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.dense_layer = nn.Linear(config.input_dim, config.input_dim)

    def forward(self, ref, mut, mask=None):
        q = mut
        k = ref
        v = ref
        attn_output, _ = self.multihead_attn(q, k, v, key_padding_mask=mask)
        out = self.dense_layer(attn_output)
        out = self.dropout(out)
        return out

class Block_CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = config.norm_fn(config.input_dim)
        self.norm2 = config.norm_fn(config.input_dim)
        self.attn = CrossAttention(config)
        self.mlp = MLP(config)

    def forward(self, ref, mut, mask=None):
        n1 = self.norm1(mut)
        mut = mut + self.attn(ref, n1, mask)
        n2 = self.norm2(mut)
        mut = mut + self.mlp(n2)
        return mut

#-----------Self Attention-------------#

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=config.emb_dim, 
                                                    num_heads=config.num_heads, 
                                                    dropout=config.dropout, 
                                                    batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.dense_layer = nn.Linear(config.emb_dim, config.emb_dim)
    
    def forward(self, x, pad_mask=None):
        attn_output, _ = self.multihead_attn(x, x, x, key_padding_mask=pad_mask)
        out = self.dense_layer(attn_output)
        out = self.dropout(out)
        return out

class Block_SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = config.norm_fn(config.emb_dim)  # RMS normalization
        self.norm2 = config.norm_fn(config.emb_dim)
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, pad_mask = None):
        n1 = self.norm1(x)
        x = x + self.attn(n1, pad_mask = pad_mask)
        n2 = self.norm2(x)
        x = x + self.mlp(n2)
        return x

########################################
# Somatt 
########################################

# class Somatt(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.blocks = nn.ModuleList([Block_SelfAttention(config) for _ in range(config.n_layer)])
#         self.classifier = nn.Linear(config.gene_emb_dim, config.n_labels)
#         self.norm = config.norm_fn(config.gene_emb_dim)

#     def forward(self, x, positions=None, return_embedding=False):
#         pad_mask = (x.sum(dim=-1) == 0)
        
#         for block in self.blocks:
#             x = block(x, positions, pad_mask=pad_mask)  # [B, G, gene_emb_dim]

#         valid_mask = (~pad_mask).unsqueeze(-1).float()  # [B, G, 1]
#         sum_embeddings = (x * valid_mask).sum(dim=1)  # [B, gene_emb_dim]
#         count = valid_mask.sum(dim=1)  # [B, 1]
#         tumor_emb = sum_embeddings / (count + 1e-8)  # [B, gene_emb_dim]

#         logits = self.classifier(tumor_emb)  # [B, num_labels]

#         if return_embedding:
#             return logits, tumor_emb
#         return logits