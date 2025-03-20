import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import pandas as pd

class Dataset_MutationList(Dataset):
    def __init__(self, data_df, embeddings, device):
        load_str = lambda x: list(map(int,x.split(',')))
        self.labels = torch.tensor(data_df['int_label'].values,dtype=torch.long)
        self.data_ = [load_str(i) for i in data_df['idxs'].values]
        self.embeddings = embeddings
        self.device = device

    def __len__(self):
        return len(self.data_)

    def __getitem__(self, idx):
        idxs = self.data_[idx]
        emb = torch.stack([torch.tensor(self.embeddings[i],dtype=torch.float32) for i in idxs if all(self.embeddings[i]!=0)])
        return emb.to(self.device), self.labels[idx].to(self.device)
    
class Dataset_Binary(Dataset):
    def __init__(self,data_df,label_col,device):
        load_str = lambda x: list(map(int,x.split(',')))
        self.labels = torch.tensor(data_df[label_col].values,dtype=torch.long)
        self.data_ = [load_str(i) for i in data_df['idxs_binary'].values]
        print(f'{len(self.data_)} samples')
        self.device = device
    
    def __len__(self):
        return len(self.data_)
    
    def __getitem__(self,idx):
        idxs = self.data_[idx]
        binary = torch.zeros(1448)
        binary[idxs] = 1
        return binary.to(self.device), self.labels[idx].to(self.device)
    
class Dataset_TopN_bin(Dataset):
    def __init__(self,data_df,device, n=10):
        load_str = lambda x: list(map(int,x.split(',')))
        self.data_ = [load_str(i) for i in data_df['idxs'].values]
        unique_idxs = set(self.data.flatten())
        self.data_order = {i:ni for ni,i in enumerate(unique_idxs)}
        self.labels = torch.tensor(data_df['int_label'].values,dtype=torch.long)
        self.device = device
        self.n = n
    
    def __len__(self):
        return len(self.data_)
    
    def __getitem__(self,idx):
        idxs = self.data_[idx]
        idxs = [self.data_order[i] for i in idxs]
        binary = torch.zeros(self.n)
        binary[idxs] = 1
        return binary.to(self.device), self.labels[idx].to(self.device)
    
class Dataset_TopN_emb(Dataset):
    def __init__(self, data_df_emb, mut_embeddings, ref_embeddings, tumors, device, n = 10, flat = True, pca=False):
        load_str = lambda x: list(map(int,x.split(',')))
        flatten = lambda x: [i for j in x for i in j]

        self.data = [load_str(i) for i in data_df_emb['idxs'].values]
        self.labels = torch.tensor(data_df_emb['int_label'].values,dtype=torch.long)
        self.mut_embeddings = mut_embeddings
        self.ref_embeddings = ref_embeddings
        if pca:
            pca_mut = PCA(n_components=pca)
            pca_mut.fit(mut_embeddings)
            self.mut_embeddings = pca_mut.transform(mut_embeddings)
            pca_ref = PCA(n_components=pca)
            pca_ref.fit(ref_embeddings)
            self.ref_embeddings = pca_ref.transform(ref_embeddings)

        self.flat = flat

        self.ref_map = {i[5]:i[4] for j in tumors.values() for i in j for k in i}
        unique_idxs = list(set(flatten(self.data)))
        print(f'{len(unique_idxs)} unique mutations')
        genes = list(set([self.ref_map[i] for i in unique_idxs]))
        self.n = len(genes)
        self.gene_order = {i:j for j,i in enumerate(genes)}
        self.base_emb = torch.stack([torch.tensor(self.ref_embeddings[i],dtype=torch.float32) for i in genes])
        print(f'{self.n} genes')
        self.device = device
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        idxs = self.data[idx]
        # print(idxs)
        
        emb = self.base_emb.clone()
        order = [self.gene_order[self.ref_map[i]] for i in idxs if all(self.mut_embeddings[i]!=0)] 
        # print(order)
        emb_ = torch.stack([torch.tensor(self.mut_embeddings[i],dtype=torch.float32) for i in idxs if all(self.mut_embeddings[i]!=0)])
        emb[order] = emb_
        # print(emb)
        if self.flat: emb = emb.flatten()
        return emb.to(self.device), self.labels[idx].to(self.device)

class Dataset_Fusion_MutationList_Binary(Dataset):
    def __init__(self, data_bin, data_emb, data_bin_label_map, data_emb_label_map, embeddings, device):
        load_str = lambda x: list(map(int,x.split(',')))
        self.data_emb_ = [load_str(i) for i in data_emb['idxs'].values]
        self.data_bin_ = [load_str(i) for i in data_bin['idxs'].values]
        self.barcode_emb = data_emb['barcode'].values
        self.map_barcode_to_binIdx = {i:ni for ni,i in enumerate(data_bin['barcode'].values)}
        self.labels_emb = data_emb['int_label'].values
        self.labels_bin = data_bin['int_label'].values
        # map the emb int labels to bin int labels
        data_emb_label_map = {v:k for k,v in data_emb_label_map.items()}
        data_bin_label_map = {v:k for k,v in data_bin_label_map.items()}
        self.map_binIntLabel_to_embIntLabel = {}
        for str_label in data_emb['str_label'].values:
            emb_int = data_emb_label_map[str_label]
            bin_int = data_bin_label_map[str_label]
            self.map_binIntLabel_to_embIntLabel[bin_int] = emb_int
        self.embeddings = embeddings
        self.device = device

    def __len__(self):
        return len(self.data_emb_)

    def __getitem__(self, idx_emb):
        #emb
        idxs_emb = self.data_emb_[idx_emb]
        emb = torch.stack([torch.tensor(self.embeddings[i],dtype=torch.float32) for i in idxs_emb if all(self.embeddings[i]!=0)])

        #binary
        barcode_emb = self.barcode_emb[idx_emb]
        idx_bin = self.map_barcode_to_binIdx[barcode_emb]
        idxs_bin = self.data_bin_[idx_bin]
        binary = torch.zeros(1448)
        binary[idxs_bin] = 1

        #label
        label_emb = self.labels_emb[idx_emb]
        label_bin = self.labels_bin[idx_bin]
        label_bin = self.map_binIntLabel_to_embIntLabel[label_bin]
        assert label_emb == label_bin
        label = torch.tensor(label_emb,dtype=torch.long)

        return emb.to(self.device), binary.to(self.device), label.to(self.device)

class Dataset_Assay(Dataset):
    def __init__(self, data_df, label_col, mut_embeddings, ref_embeddings, tumors, assays, device, pos=True):
        load_str = lambda x: list(map(int, x.split(',')))
        flatten = lambda x: [i for j in x for i in j]
        self.data = [load_str(i) for i in data_df['idxs'].values]
        self.labels = torch.tensor(data_df[label_col].values, dtype=torch.long)
        self.assay_labels = data_df['assay'].values
        self.assays = assays
        self.mut_embeddings = mut_embeddings
        self.ref_embeddings = ref_embeddings
        self.ref_map = {i[5]: i[4] for j in tumors.values() for i in j for k in i}
        self.device = device
        self.pos = pos
        print(f'{len(self.data)} samples')
        print(f'{data_df["patient_id"].nunique()} unique patients')
        print(f'{data_df[label_col].nunique()} labels')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mutation_indices = self.data[idx]  # e.g., [mut1, mut2, mut3]
        assay_id = self.assay_labels[idx]
        assay_gene_idxs = self.assays[assay_id]  # e.g., [1, 2, 3, 4, 5]
        ref_emb = torch.stack([
            torch.tensor(self.ref_embeddings[gene_idx], dtype=torch.float32)
            for gene_idx in assay_gene_idxs
        ])  
        mut_emb = torch.stack([
            torch.tensor(self.mut_embeddings[mut_idx], dtype=torch.float32)
            for mut_idx in mutation_indices
        ])  
        emb = torch.cat([ref_emb, mut_emb], dim=0)  
        mut_assay_idxs = [self.ref_map[mut_idx] for mut_idx in mutation_indices]  
        combined_assay_idxs = assay_gene_idxs + mut_assay_idxs  
        assay_idxs_tensor = torch.tensor(combined_assay_idxs, dtype=torch.long).to(self.device)
        if not self.pos:
            assay_idxs_tensor = None
        return emb.to(self.device), assay_idxs_tensor, self.labels[idx].to(self.device)

class Dataset_Classification_MutationsOnly(Dataset):
    def __init__(self, data_df, label_col, mut_embeddings, tumors, device, pos=True):
        load_str = lambda x: list(map(int, x.split(',')))
        self.data = [load_str(i) for i in data_df['idxs'].values]
        self.labels = torch.tensor(data_df[label_col].values, dtype=torch.long)
        self.mut_embeddings = mut_embeddings
        self.ref_map = {i[5]: i[4] for j in tumors.values() for i in j for k in i}
        self.device = device
        self.pos = pos
        print(f'{len(self.data)} samples')
        print(f'{data_df["patient_id"].nunique()} unique patients')
        print(f'{data_df[label_col].nunique()} labels')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mutation_indices = self.data[idx]  
        mut_emb = torch.stack([torch.tensor(self.mut_embeddings[mut_idx], dtype=torch.float32) for mut_idx in mutation_indices])  
        mut_idxs = [self.ref_map[mut_idx] for mut_idx in mutation_indices]  # e.g., [2, 2, 3]
        mut_idxs_tensor = torch.tensor(mut_idxs, dtype=torch.long).to(self.device)
        if not self.pos:
            mut_idxs_tensor = None
        return mut_emb.to(self.device), mut_idxs_tensor, self.labels[idx].to(self.device)

class Dataset_Survival_MutationsOnly(Dataset):
    def __init__(self, data_df, mut_embeddings, tumors, device):
        load_str = lambda x: list(map(int, x.split(',')))
        assert data_df['time'].isna().sum() == 0, 'time column contains nan values'
        assert data_df['time'].min() > 0, 'time column contains non-positive values'       
        print(len(data_df),'samples')
        print(data_df['patient_id'].nunique(),'unique patients')
        print(data_df['CANCER_TYPE'].nunique(),'cancer types')
        print(data_df['CANCER_TYPE_DETAILED'].nunique(),'detailed cancer types')
        self.data = [load_str(i) for i in data_df['idxs'].values]
        self.times = torch.tensor(data_df['time'].values, dtype=torch.long)
        self.events = torch.tensor(data_df['censor'].values, dtype=torch.long)
        self.mut_embeddings = mut_embeddings
        self.ref_map = {i[5]:i[4] for j in tumors.values() for i in j for k in i}
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mutation_indices = self.data[idx]
        mut_emb = torch.stack([torch.tensor(self.mut_embeddings[i], dtype=torch.float32) for i in mutation_indices])
        mut_idxs = [self.ref_map[i] for i in mutation_indices]
        mut_idxs_tensor = torch.tensor(mut_idxs, dtype=torch.long).to(self.device)
        return mut_emb.to(self.device), mut_idxs_tensor, self.times[idx].to(self.device), self.events[idx].to(self.device)

    
def custom_collate_DNN_classifier(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = torch.stack(data)
    labels = torch.stack(labels)
    return data, labels

def custom_collate_DNN_survival(batch):
    data = [item[0] for item in batch]
    times = [item[1] for item in batch]
    events = [item[2] for item in batch]
    data = torch.stack(data)
    times = torch.stack(times)
    events = torch.stack(events)
    return data, times, events

def custom_collate_assay_fusion(batch):
    somatt = [item[0] for item in batch]
    assay = [item[1] for item in batch]
    dnn = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    somatt = pad_sequence(somatt, batch_first=True, padding_value=0)
    assay = pad_sequence(assay, batch_first=True, padding_value=assay[0][0])
    dnn = torch.stack(dnn)
    labels = torch.stack(labels)
    return (somatt, assay, dnn), labels

def custom_collate_assay_survival_fusion(batch):
    somatt = [item[0] for item in batch]
    assay = [item[1] for item in batch]
    dnn = [item[2] for item in batch]
    times = [item[3] for item in batch]
    events = [item[4] for item in batch]

    somatt = pad_sequence(somatt, batch_first=True, padding_value=0)
    assay = pad_sequence(assay, batch_first=True, padding_value=assay[0][0])
    dnn = torch.stack(dnn)
    times = torch.stack(times)
    events = torch.stack(events)
    return (somatt, assay, dnn), times, events
    
class Dataset_DNN_Classifier(Dataset):
    def __init__(self, data_df, device):
        label_col = 'CANCER_TYPE_INT'
        self.X = torch.tensor(data_df.drop(columns=[label_col,'barcode','patient_id']).values,dtype=torch.float32)
        self.labels = torch.tensor(list(map(int,data_df[label_col].values)),dtype=torch.long)
        self.device = device
        print(f'{data_df["patient_id"].nunique()} unique patients')
        print(f'{data_df[label_col].nunique()} labels')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        input = self.X[idx]
        return input.to(self.device), self.labels[idx].to(self.device)
    
class Dataset_Assay_Classification_Fusion(Dataset):
    def __init__(self, data_dfs, mut_embeddings, ref_embeddings, tumors, assays, device, label_col = 'CANCER_TYPE_INT'):
        data_df_dnn, data_df = data_dfs 
        load_str = lambda x: list(map(int,x.split(',')))
        flatten = lambda x: [i for j in x for i in j]
        self.data = [load_str(i) for i in data_df['idxs'].values]
        self.labels = torch.tensor(data_df[label_col].values,dtype=torch.long)
        self.assay_labels = data_df['assay'].values
        self.assays = assays
        self.mut_embeddings = mut_embeddings
        self.ref_embeddings = ref_embeddings
        self.ref_map = {i[5]:i[4] for j in tumors.values() for i in j for k in i}
        self.device = device
        print(f'{len(self.data)} samples')
        print(f'{data_df["patient_id"].nunique()} unique patients')
        print(f'{data_df[label_col].nunique()} labels')

        self.dnn_X = torch.tensor(data_df_dnn.drop(columns=[label_col,'barcode','patient_id']).values,dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        idxs = self.data[idx]
        assay_id = self.assay_labels[idx]
        assay_idxs = self.assays[assay_id]
        gene_order = {i:j for j,i in enumerate(assay_idxs)}
        order = [gene_order[self.ref_map[i]] for i in idxs] 
        emb = torch.stack([torch.tensor(self.ref_embeddings[i],dtype=torch.float32) for i in assay_idxs])
        emb_ = torch.stack([torch.tensor(self.mut_embeddings[i],dtype=torch.float32) for i in idxs])
        emb[order] = emb_
        dnn_input = self.dnn_X[idx].to(self.device)
        return emb.to(self.device), torch.tensor(assay_idxs,dtype=torch.long).to(self.device), dnn_input, self.labels[idx].to(self.device)
    
class Dataset_DNN_Survival(Dataset):
    def __init__(self, data_df, device):
        assert data_df['time'].isna().sum() == 0, 'time column contains nan values'
        assert data_df['time'].min() > 0, 'time column contains non-positive values'
        self.X = torch.tensor(data_df.drop(columns=['time','censor','barcode','patient_id']).values,dtype=torch.float32)
        self.times = torch.tensor(data_df['time'].values,dtype=torch.long)
        self.events = torch.tensor(data_df['censor'].values,dtype=torch.long)
        self.device = device
        print(f'{data_df["patient_id"].nunique()} unique patients')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        input = self.X[idx]
        return input.to(self.device), self.times[idx].to(self.device), self.events[idx].to(self.device)

class Dataset_Assay_Survival_Fusion(Dataset):
    def __init__(self, data_dfs, mut_embeddings, ref_embeddings, tumors, assays, device):
        data_df_dnn, data_df = data_dfs
        load_str = lambda x: list(map(int,x.split(',')))
        flatten = lambda x: [i for j in x for i in j]
        assert data_df['time'].isna().sum() == 0, 'time column contains nan values'
        assert data_df['time'].min() > 0, 'time column contains non-positive values'       
        print(len(data_df),'samples')
        print(data_df['patient_id'].nunique(),'unique patients')
        print(data_df['CANCER_TYPE'].nunique(),'cancer types')
        print(data_df['CANCER_TYPE_DETAILED'].nunique(),'detailed cancer types')
        self.data = [load_str(i) for i in data_df['idxs'].values]
        self.times = torch.tensor(data_df['time'].values,dtype=torch.long)
        self.events = torch.tensor(data_df['censor'].values,dtype=torch.long)
        self.assay_labels = data_df['assay'].values
        self.assays = assays
        self.mut_embeddings = mut_embeddings
        self.ref_embeddings = ref_embeddings
        self.ref_map = {i[5]:i[4] for j in tumors.values() for i in j for k in i}
        self.device = device

        self.X = torch.tensor(data_df.drop(columns=['time','censor','barcode','patient_id']).values,dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        idxs = self.data[idx]
        assay_id = self.assay_labels[idx]
        assay_idxs = self.assays[assay_id]
        gene_order = {i:j for j,i in enumerate(assay_idxs)}
        order = [gene_order[self.ref_map[i]] for i in idxs] 
        emb = torch.stack([torch.tensor(self.ref_embeddings[i],dtype=torch.float32) for i in assay_idxs])
        emb_ = torch.stack([torch.tensor(self.mut_embeddings[i],dtype=torch.float32) for i in idxs])
        emb[order] = emb_
        
        dnn_input = self.X[idx].to(self.device)
        return emb.to(self.device), torch.tensor(assay_idxs,dtype=torch.long).to(self.device), dnn_input, self.times[idx].to(self.device), self.events[idx].to(self.device)

class Dataset_Assay_Survival(Dataset):
    def __init__(self, data_df, mut_embeddings, ref_embeddings, tumors, assays, device):
        load_str = lambda x: list(map(int, x.split(',')))
        assert data_df['time'].isna().sum() == 0, 'time column contains nan values'
        assert data_df['time'].min() > 0, 'time column contains non-positive values'       
        print(len(data_df),'samples')
        print(data_df['patient_id'].nunique(),'unique patients')
        print(data_df['CANCER_TYPE'].nunique(),'cancer types')
        print(data_df['CANCER_TYPE_DETAILED'].nunique(),'detailed cancer types')
        self.data = [load_str(i) for i in data_df['idxs'].values]
        self.times = torch.tensor(data_df['time'].values, dtype=torch.long)
        self.events = torch.tensor(data_df['censor'].values, dtype=torch.long)
        self.assay_labels = data_df['assay'].values
        self.assays = assays
        self.mut_embeddings = mut_embeddings
        self.ref_embeddings = ref_embeddings
        self.ref_map = {i[5]:i[4] for j in tumors.values() for i in j for k in i}
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mutation_indices = self.data[idx]
        assay_id = self.assay_labels[idx]
        assay_gene_idxs = self.assays[assay_id]
        ref_emb = torch.stack([torch.tensor(self.ref_embeddings[i], dtype=torch.float32) for i in assay_gene_idxs])
        mut_emb = torch.stack([torch.tensor(self.mut_embeddings[i], dtype=torch.float32) for i in mutation_indices])
        emb = torch.cat([ref_emb, mut_emb], dim=0)
        mut_assay_idxs = [self.ref_map[i] for i in mutation_indices]
        combined_assay_idxs = assay_gene_idxs + mut_assay_idxs
        assay_idxs_tensor = torch.tensor(combined_assay_idxs, dtype=torch.long).to(self.device)
        return emb.to(self.device), assay_idxs_tensor, self.times[idx].to(self.device), self.events[idx].to(self.device)

    
class Dataset_Binary_Survival(Dataset):
    def __init__(self,data_df,device):
        assert data_df['time'].isna().sum() == 0, 'time column contains nan values'
        assert data_df['time'].min() > 0, 'time column contains non-positive values'
        load_str = lambda x: list(map(int,x.split(',')))
        print(len(data_df),'samples')
        print(data_df['patient_id'].nunique(),'unique patients')
        print(data_df['CANCER_TYPE'].nunique(),'cancer types')
        print(data_df['CANCER_TYPE_DETAILED'].nunique(),'detailed cancer types')
        self.times = torch.tensor(data_df['time'].values,dtype=torch.long)
        self.events = torch.tensor(data_df['censor'].values,dtype=torch.long)
        self.data_ = [load_str(i) for i in data_df['idxs_binary'].values]
        self.device = device
    
    def __len__(self):
        return len(self.data_)
    
    def __getitem__(self,idx):
        idxs = self.data_[idx]
        binary = torch.zeros(1448)
        binary[idxs] = 1
        return binary.to(self.device), self.times[idx].to(self.device), self.events[idx].to(self.device)

def custom_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = pad_sequence(data, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return data, labels

def custom_collate_assay(batch):
    data = [item[0] for item in batch]
    labels = [item[2] for item in batch]
    data = pad_sequence(data, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    if batch[0][1] is not None:
        assay = [item[1] for item in batch]
        assay = pad_sequence(assay, batch_first=True, padding_value=assay[0][0])
    else: assay = None
    return data, assay, labels

def custom_collate_assay_survival(batch):
    data = [item[0] for item in batch]
    assay = [item[1] for item in batch]
    times = [item[2] for item in batch]
    events = [item[3] for item in batch]

    data = pad_sequence(data, batch_first=True, padding_value=0)
    assay = pad_sequence(assay, batch_first=True, padding_value=assay[0][0])
    times = torch.stack(times)
    events = torch.stack(events)
    return data, assay, times, events