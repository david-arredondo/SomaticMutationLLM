import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from sklearn.decomposition import PCA

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
    def __init__(self,data_df,device):
        load_str = lambda x: list(map(int,x.split(',')))
        self.labels = torch.tensor(data_df['int_label'].values,dtype=torch.long)
        self.data_ = [load_str(i) for i in data_df['idxs'].values]
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

def custom_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = pad_sequence(data, batch_first=True, padding_value=float('-inf'))
    labels = torch.stack(labels)
    return data, labels
