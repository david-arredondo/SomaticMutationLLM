import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class Dataset_MutationList(Dataset):
    def __init__(self, data, labels, embeddings, device):
        load_str = lambda x: list(map(int,x.split(',')))
        self.data_ = [load_str(i) for i in data]
        self.labels = labels
        self.embeddings = embeddings
        self.device = device

    def __len__(self):
        return len(self.data_)

    def __getitem__(self, idx):
        idxs = self.data_[idx]
        emb = torch.stack([torch.tensor(self.embeddings[i],dtype=torch.float32) for i in idxs])
        return emb.to(self.device), self.labels[idx].to(self.device)

def custom_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = pad_sequence(data, batch_first=True, padding_value=float('-inf'))
    labels = torch.stack(labels)
    return data, labels
