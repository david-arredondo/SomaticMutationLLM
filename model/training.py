#!/usr/bin/env python
# coding: utf-8

from custom_dataset import Dataset_Assay, Dataset_Assay_Survival, custom_collate_assay, custom_collate_assay_survival, Dataset_Survival_MutationsOnly, Dataset_Classification_MutationsOnly
from models import *
from saveAndLoad import *

class Config_Att:
    n_layer: int = 3
    input_dim: int = 640 #1152 esmC #1536 esm3 #640 esm2
    dropout: float = 0.0
    bias: bool = False
    n_labels: int
    pooling : str = 'mean'
    norm_fn: nn.Module = nn.LayerNorm
    max_len : int = 1433
    position_embedding: bool = False
    num_heads: int = 1

def train_cancer_type(modelClass, configClass, dataset, data_df_emb, saveName, n_labels, n_folds=None, test_size = None, num_epochs = 15, lr = .0001, device = 'cuda:1'):

    folds = getPatientGroupedLoaders(dataset, data_df_emb, n_folds=n_folds, test_size = test_size, batch_size = 100, collate=custom_collate_assay)

    config_att = configClass()
    config_att.n_labels = n_labels

    print(data_df_emb["CANCER_TYPE_INT"].unique())
    print("Max label:", data_df_emb["CANCER_TYPE_INT"].max())
    print("Min label:", data_df_emb["CANCER_TYPE_INT"].min())
    
    criterion = nn.CrossEntropyLoss()

    for i, (train_loader, test_loader) in enumerate(folds):
        model = modelClass(config_att)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print(f'\nFOLD {i+1}')
        if saveName is not None: saveName = './best_models/' + saveName.split('.')[0] + f'_fold{i+1}.pt'
        train_assay(model,num_epochs,train_loader,test_loader, criterion, optimizer, saveName = saveName)


def train_survival(modelClass, configClass, dataset, data_df_emb, saveName, n_folds = None, test_size = None, num_epochs = 5, lr = .0001, device = 'cuda:1'):

    folds = getPatientGroupedLoaders(dataset, data_df_emb, n_folds=n_folds, test_size = test_size, batch_size = 100, collate=custom_collate_assay_survival)

    config_att = configClass()
    config_att.n_labels = 1

    criterion = negative_log_partial_likelihood

    for i, (train_loader, test_loader) in enumerate(folds):
        model = modelClass(config_att)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print(f'\nFOLD {i+1}')
        if saveName is not None: saveName = './best_models/' + saveName.split('.')[0] + f'_fold{i+1}.pt'
        train_assay_survival(model,num_epochs,train_loader,test_loader, criterion, optimizer, saveName = saveName)

def get_df(data_file, suffix = ''):
    data_path = '../labeled_data/' + data_file
    data_df_emb = pd.read_csv(data_path)
    saveName =  f'model_sha_{data_file.split(".")[0]}_{suffix}.pt'
    return data_df_emb, saveName