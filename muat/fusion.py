import sys
if '../model/' not in sys.path: sys.path.append('../model/')
from models import negative_log_partial_likelihood, getPatientGroupedLoaders, c_index, SoMatt_LateFusion
from custom_dataset import Dataset_Assay_Classification_Fusion, Dataset_Assay_Survival_Fusion, custom_collate_assay_survival_fusion, custom_collate_assay_fusion
from dnn import train_classifier, train_survival
import torch.optim as optim
import torch.nn as nn

class Config_Fusion:
    n_layer: int = 3
    input_dim: int = 640
    fusion_dim: int 
    dropout: float = 0.0
    bias: bool = False
    n_labels: int
    pooling : str = 'mean'
    norm_fn: nn.Module = nn.LayerNorm
    max_len : int = 1433
    position_embedding: bool = False
    num_heads: int = 1

def run_fusion(class_data,
            dfs,
            mut_embeddings,
            ref_embeddings,
            tumors,
            assays,
            fusion_dim = 150,
            survival=False,
            device = 'cuda:0',
            num_epochs = 2,
            n_folds = None,
            test_size = .2,
            lr = .0001,
            saveName = None
            ):
    
    df, somatt_df = dfs    

    if survival:
        output_dim = 1 
        criterion = negative_log_partial_likelihood  
        train_fn = train_survival
        to_merge = ['barcode','time','censor', 'patient_id']
        dataset_class = Dataset_Assay_Survival_Fusion 
        collate_fn = custom_collate_assay_survival_fusion
    else:
         output_dim = max(class_data['CANCER_TYPE_INT'].unique()) + 1
         criterion = nn.CrossEntropyLoss()
         train_fn = train_classifier
         to_merge =  ['barcode','CANCER_TYPE_INT', 'patient_id']
         dataset_class = Dataset_Assay_Classification_Fusion
         collate_fn = custom_collate_assay_fusion

    df_merged = df.merge(
        class_data[to_merge], 
        left_index=True, 
        right_on='barcode'
        )
    
    dfs = (df_merged, somatt_df)
    dataset = dataset_class(dfs, mut_embeddings, ref_embeddings, tumors, assays, device)        
    folds = getPatientGroupedLoaders(dataset, df_merged, n_folds=n_folds, test_size = test_size, batch_size = 500, collate=collate_fn)

    for i, (train_loader, test_loader) in enumerate(folds):
        config = Config_Fusion()
        config.fusion_dim = fusion_dim
        config.n_labels = output_dim
        model = SoMatt_LateFusion(config)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print(f'\nFOLD {i+1}')
        if saveName is not None: saveName = './best_models/' + saveName.split('.')[0] + f'_fold{i+1}.pt'
        train_fn(model,num_epochs,train_loader,test_loader, criterion, optimizer, saveName = None)