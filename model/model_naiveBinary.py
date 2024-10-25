#!/usr/bin/env python
# coding: utf-8

from custom_dataset import Dataset_Binary, Dataset_Binary_Survival
from models import *
from saveAndLoad import *

tumors = pickleLoad('../aa/tumors_binary.pkl')
device = 'cuda:1'

# LOAD DATA
data_dir = '../labeled_data/'
labeled_data = os.listdir(data_dir)
for i,ni in sorted(zip(labeled_data,range(len(labeled_data)))):
    for t in ['CANCER_TYPE_INT','CANCER_TYPE_DETAILED_INT']:
        if not os.path.isdir(data_dir+i):
            print(i)
            data_df_bin = pd.read_csv(data_dir+i)
            nlabels = len(data_df_bin[t].unique())
            print('n labels:',nlabels)

            dataset = Dataset_Binary(data_df_bin, t, device)
            train_loader, test_loader = getTrainTestLoaders(dataset, batch_size = 750)

            class Config:
                input_dim: int = 1448
                bias: bool = False
                n_labels: int = 17

            print('n labels:',nlabels)
            config = Config()
            config.n_labels = nlabels

            model = MLPClassifier(config)
            model.to(device)

            num_epochs = 10
            learning_rate = 0.001
            saveName = f'./best_models/model_binary_{t}_{i.split(".")[0]}.pt'
            print(saveName)
            train(model,num_epochs,train_loader,test_loader,saveName=saveName)
        
## SURVIVAL
data_df_bin = pd.read_csv(data_dir+'data_0_00percentMinCancerType.csv')
dataset_binary_survival = Dataset_Binary_Survival(data_df_bin, device, ratio=1)
train_loader_survival, test_loader_survival = getTrainTestLoaders(dataset_binary_survival, batch_size = 500)

## BINARY SURVIVAL MODEL
class Config_Bin:
    input_dim: int = 1448
    bias: bool = False
    n_labels: int = 1

config = Config_Bin()
model = MLPClassifier(config)
model.to(device)
num_epochs = 10

optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
criterion = negative_log_partial_likelihood
saveName = './best_models/model_binary_survival.pt'
print(saveName)
train_binary_survival(model,num_epochs,train_loader_survival,test_loader_survival, criterion, optimizer, saveName=saveName)