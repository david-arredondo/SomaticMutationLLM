#!/usr/bin/env python
# coding: utf-8

from custom_dataset import Dataset_Assay, Dataset_Assay_Survival, custom_collate_assay, custom_collate_assay_survival
from models import *
from saveAndLoad import *

mut_embeddings = np.load('../aa/canonical_mut_average_embeddings_esm2.npy')
ref_embeddings = np.load('../aa/canonical_ref_embeddings_esm2.npy')
tumors = pickleLoad('../aa/tumors.pkl')
assays = pickleLoad('../aa/assays.pkl')
device = 'cuda:0'

# LOAD DATA
data_dir = '../labeled_data/'
labeled_data = os.listdir(data_dir)
for i,ni in sorted(zip(labeled_data,range(len(labeled_data)))):
    if ('BINARY' not in i) and not os.path.isdir(data_dir+i):
        print(i)
        data_df_emb = pd.read_csv(data_dir+i)

        nlabels = len(data_df_emb['int_label'].unique())
        print('n labels:',nlabels)

        dataset = Dataset_Assay(data_df_emb, mut_embeddings, ref_embeddings, tumors, assays, device)
        train_loader, test_loader = getTrainTestLoaders(dataset, batch_size = 500, collate=custom_collate_assay)

        ## CANCER TYPE
        class Config_Att:
            n_layer: int = 3
            input_dim: int = 640
            dropout: float = 0.0
            bias: bool = False
            n_labels: int
            pooling : str = 'mean'
            norm_fn: nn.Module = nn.LayerNorm
            max_len : int = 1448
            position_embedding: bool = False
            num_heads: int = 8

        config_att = Config_Att()
        config_att.n_labels = nlabels

        model = Classifier(config_att)
        model.to(device)

        num_epochs = 7
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=.001)
        saveName = './best_models/model_mha_'+i.split('.')[0]+'.pt'
        print(saveName)
        train_assay(model,num_epochs,train_loader,test_loader, criterion, optimizer, saveName=saveName)

## SURVIVAL
data_df_emb = pd.read_csv(data_dir+'data_CANCER_TYPE_0_00percentMinCancerType.csv')
dataset = Dataset_Assay_Survival(data_df_emb, mut_embeddings, ref_embeddings, tumors, assays, device)
train_loader, test_loader = getTrainTestLoaders(dataset, batch_size = 500, collate=custom_collate_assay_survival)

class Config_Att:
    n_layer: int = 3
    input_dim: int = 640
    dropout: float = 0.0
    bias: bool = False
    n_labels: int = 1
    pooling : str = 'mean'
    norm_fn: nn.Module = nn.LayerNorm
    max_len : int = 1448
    position_embedding: bool = False
    num_heads: int = 8

config_att = Config_Att()

model = Classifier(config_att)
model.to(device)

num_epochs = 5
optimizer = optim.Adam(model.parameters(), lr=.0001)
criterion = negative_log_partial_likelihood
saveName = './best_models/model_mha_survival.pt'
print(saveName)
train_assay_survival(model,num_epochs,train_loader,test_loader, criterion, optimizer, saveName=saveName)



