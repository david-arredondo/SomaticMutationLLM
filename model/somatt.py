import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from functools import partial
from models import *
from saveAndLoad import *
from functools import partial
import math

class Dataset_Somatt(Dataset):
    def __init__(self, data_df, mut_embeddings, ref_embeddings, tumors, assays, device):
        self.data_df = data_df
        self.mutations = data_df['idxs'].values
        self.cancer_type = data_df['CANCER_TYPE_INT'].values
        self.cancer_type_detailed = data_df['CANCER_TYPE_DETAILED_INT'].values
        self.assay = data_df['assay'].values
        self.assays = assays
        self.time = data_df['time'].values
        self.age = data_df['age'].values
        self.event = data_df['censor'].values
        self.sex = data_df['sex_INT'].values
        self.race = data_df['race_INT'].values
        self.ref_af = data_df['ref_af'].values
        self.mut_af = data_df['mut_af'].values
        self.focal_cna_gene_id = data_df['focal_cna_gene_id'].values
        self.focal_cna_alteration = data_df['focal_cna_alteration_INT'].values
        self.arm = data_df['arm_INT'].values
        self.start = data_df['start'].values
        self.end = data_df['end'].values
        self.seg_mean = data_df['seg_mean'].values
        self.mut_embeddings = mut_embeddings
        self.ref_embeddings = ref_embeddings
        self.ref_map = {i[5]: i[4] for j in tumors.values() for i in j for k in i}
        self.device = device
        print(f'{len(data_df)} samples')
        print(f'{data_df["patient_id"].nunique()} unique patients')

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):

        cancer = self.cancer_type[idx]
        cancer_detailed = self.cancer_type_detailed[idx]
        sex = self.sex[idx]
        age = self.age[idx]
        race = self.race[idx]
        time = self.time[idx]
        event = self.event[idx]

        long_tensor = lambda x: torch.tensor(x, dtype=torch.long)
        float_tensor = lambda x: torch.tensor(x, dtype=torch.float32)

        #mutation embeddings
        mutation_indices = self.mutations[idx]  # e.g., [mut1, mut2, mut3]
        mut_emb = torch.stack([
            torch.tensor(self.mut_embeddings[mut_idx], dtype=torch.float32)
            for mut_idx in mutation_indices
        ])  
        mut_id = [self.ref_map[mut_idx] for mut_idx in mutation_indices]
        mut_af = torch.tensor(self.mut_af[idx],dtype=torch.float32)

        #corresponding reference embeddings
        ref_indices = mut_id  # e.g., [ref1, ref2, ref3]
        ref_emb = torch.stack([
            torch.tensor(self.ref_embeddings[ref_idx], dtype=torch.float32)
            for ref_idx in ref_indices
        ])
        ref_af = torch.tensor(self.ref_af[idx],dtype=torch.float32)

        #rest of reference embeddings in assay
        assay_id = self.assay[idx]
        assay_gene_idxs = self.assays[assay_id]  # e.g., [1, 2, 3, 4, 5]
        assay_gene_idxs = [x for x in assay_gene_idxs if ((x not in mut_id) and (x not in ref_indices))]
        if len(assay_gene_idxs) > 0:
            ref_emb = torch.cat([
                ref_emb,
                torch.stack([
                torch.tensor(self.ref_embeddings[gene_idx], dtype=torch.float32)
                for gene_idx in assay_gene_idxs
                ])
            ])
            ref_af = torch.cat([ref_af, torch.tensor([1 for _ in range(len(assay_gene_idxs))], dtype=torch.float32)])

        #combine all embeddings
        gene_id = torch.cat([torch.tensor(mut_id), torch.tensor(mut_id), torch.tensor(assay_gene_idxs)])
        gene_emb = torch.cat([mut_emb, ref_emb])
        maf = torch.cat([mut_af, ref_af])

        focal_cna_id = long_tensor(self.focal_cna_gene_id[idx])
        focal_cna_alteration = long_tensor(self.focal_cna_alteration[idx])

        if len(focal_cna_id):
            focal_cna_gene_emb = torch.stack([
                torch.tensor(self.ref_embeddings[gene_idx], dtype=torch.float32)
                for gene_idx in focal_cna_id
            ])
        else:
            emb_dim = mut_emb.size(1)
            focal_cna_gene_emb = torch.empty((0, emb_dim), dtype = torch.float32)

        seg_id = long_tensor(self.arm[idx])
        seg_start = float_tensor(self.start[idx])
        seg_end = float_tensor(self.end[idx])
        seg_mean = float_tensor(self.seg_mean[idx])

        return cancer, cancer_detailed, sex, age, race, time, event, gene_id, gene_emb, maf, focal_cna_id, focal_cna_alteration, focal_cna_gene_emb, seg_id, seg_start, seg_end, seg_mean
    
def collate_somatt(batch, config):
    stackLong =     lambda x: torch.stack([torch.tensor([item[x]], dtype = torch.long) for item in batch])
    stackFloat =    lambda x: torch.stack([torch.tensor([item[x]], dtype=torch.float32) for item in batch])
    cancer =    stackLong(0)
    cancer_d =  stackLong(1)
    sex =       stackLong(2)
    age =       stackFloat(3)
    race =      stackLong(4)
    time =      stackFloat(5)
    event =     stackLong(6)

    def pad(batch_, i, pad_val, mask = True):
        unpadded_list = [item[i] for item in batch_]
        padded_list = pad_sequence(unpadded_list, batch_first=True, padding_value=pad_val)
        return padded_list

    gene_id =                       pad(batch, 7, config.gene_id_vocab_size)
    gene_emb =                      pad(batch, 8, 0)
    maf =                           pad(batch, 9, 0)
    focal_cna_id =                  pad(batch, 10, config.gene_id_vocab_size)
    focal_cna_alteration =          pad(batch, 11, config.focal_cna_vocab_size)
    focal_cna_gene_emb =            pad(batch, 12, 0)
    # broad_cna, broad_cna_pad_mask = pad_and_mask(batch, 11, config.broad_cna_vocab_size) #broad is binary
    
    seg_id =                        pad(batch, 13, config.seg_id_vocab_size)
    seg_start =                     pad(batch, 14, 0)                      #broad is emb
    seg_end =                       pad(batch, 15, 0)        #broad is emb
    seg_mean =                      pad(batch, 16, 0)        #broad is emb

    # return (cancer, sex, age, time, event, gene_emb, maf, focal_cna, broad_cna, pad_mask) #broad is binary
    return (cancer, cancer_d, sex, age, race, time, event, gene_id, gene_emb, maf, focal_cna_id, focal_cna_alteration, focal_cna_gene_emb, seg_id, seg_start, seg_end, seg_mean) #broad is emb

class MLPProj(nn.Module):
    def __init__(self, config, input_dim, output_dim = None):
        super().__init__()
        output_dim = config.emb_dim if output_dim is None else output_dim
        self.lin1 = nn.Linear(input_dim, config.emb_dim)
        self.lin2 = nn.Linear(config.embed_dim, config.emb_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        return x

class LinProj(nn.Module):
    def __init__(self, config, input_dim, output_dim = None):
        super().__init__()
        output_dim = config.emb_dim if output_dim is None else output_dim
        self.lin1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin1(x)

class Rbf(nn.Module): #FROM ESM3
    def __init__(self, config):
        super().__init__()
        self.v_min, self.v_max, self.n_bins = config.rbf_params
    
    def rbf(self, values):
        rbf_centers = torch.linspace(
            self.v_min, self.v_max, self.n_bins, device=values.device, dtype=values.dtype
        )
        rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
        rbf_std = (self.v_max - self.v_min) / self.n_bins
        z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
        rbf_encodings =  torch.exp(-(z**2))
        return rbf_encodings
    
    def forward(self, x):
        return self.rbf(x)
    
class EmbedSurv_rbf_2heads(Rbf):
    def __init__(self, config):
        super().__init__(config)
        self.v_min, self.v_max, self.n_bins = config.rbf_params
        self.lin0 = nn.Linear(self.n_bins, config.emb_dim)
        self.lin1 = nn.Linear(self.n_bins, config.emb_dim)
        self.emb_dim = config.emb_dim  
    
    def forward(self, time, event):
        time = time.squeeze(1)
        event = event.squeeze(1)
        output = torch.empty(time.size(0), self.emb_dim, device=time.device)
        mask = event.bool()
        rbf_encodings = self.rbf(time)
        output[mask] = self.lin1(rbf_encodings[mask])
        output[~mask] = self.lin0(rbf_encodings[~mask])
        return output.unsqueeze(1)

class Somatt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([Block_SelfAttention(config) for _ in range(config.n_layer)])
        self.norm = config.norm_fn(config.emb_dim)

        self.embed_cancer_type = nn.Embedding(config.cancer_type_vocab_size, config.emb_dim)
        self.embed_cancer_type_detailed = nn.Embedding(config.cancer_type_detailed_vocab_size, config.emb_dim)
        self.sex_nan_idx = config.sex_nan_idx
        self.embed_sex = nn.Embedding(config.sex_vocab_size, config.emb_dim, padding_idx=self.sex_nan_idx)
        self.embed_age = config.lin_proj(config, input_dim = 1)
        self.race_nan_idx = config.race_nan_idx
        self.embed_race = nn.Embedding(config.race_vocab_size, config.emb_dim, padding_idx=self.race_nan_idx)
        self.embed_surv = config.embed_surv(config)
        self.embed_clin_id = nn.Embedding(config.n_clin_vars, config.emb_dim)
        maf_emb_dim = config.emb_dim if config.maf_emb_dim is None else config.maf_emb_dim
        self.embed_maf = config.lin_proj(config, input_dim = 1, output_dim = maf_emb_dim) #RBF HERE?
        self.embed_gene = config.lin_proj(config, input_dim = config.emb_dim + maf_emb_dim)
        self.embed_gene_id = nn.Embedding(config.gene_id_vocab_size+1, config.emb_dim, padding_idx=config.gene_id_vocab_size)
        self.embed_focal_cna = nn.Embedding(config.focal_cna_vocab_size+1, config.maf_emb_dim, padding_idx=config.focal_cna_vocab_size)
        self.embed_seg_id = nn.Embedding(config.seg_id_vocab_size+1, config.emb_dim, padding_idx=config.seg_id_vocab_size)
        self.embed_seg = config.lin_proj(config, input_dim = 3)                                                                            #broad is emb
        self.embed_mask = nn.Embedding(1, config.emb_dim)
        # self.embed_broad_cna = nn.Embedding(config.broad_cna_vocab_size+1, config.emb_dim, padding_idx=config.broad_cna_vocab_size)   #broad is binary                   

        self.cancer_head = nn.Linear(config.emb_dim, config.cancer_type_vocab_size)
        self.cancer_detailed_head = nn.Linear(config.emb_dim, config.cancer_type_detailed_vocab_size)
        self.sex_head = nn.Linear(config.emb_dim, config.sex_vocab_size -1)   #UNKNOWN
        self.race_head = nn.Linear(config.emb_dim, config.race_vocab_size -1) #UNKNOWN
        self.age_head = nn.Linear(config.emb_dim, 1)
        self.survival_head = nn.Linear(config.emb_dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def mean_pool_by_id(self, embs_, ids_, pad_val, pool = True):
        B,N,emb_dim = embs_.shape

        output = []
        output_ids = []
        for b in range(B):
            ids = ids_[b]
            embs = embs_[b]
            if pool:
                unique_ids, inverse_indices = torch.unique(ids, sorted=True, return_inverse=True)
                inverse_indices = inverse_indices.detach()
                num_unique = unique_ids.size(0)
                sum_emb = torch.zeros(num_unique, emb_dim, device=embs.device)
                count   = torch.zeros(num_unique, device=embs.device)
                sum_emb = sum_emb.scatter_add_(
                    0,
                    inverse_indices.unsqueeze(-1).expand(-1, emb_dim),
                    embs
                )
                count = count.scatter_add_(
                    0,
                    inverse_indices,
                    torch.ones_like(inverse_indices, dtype=torch.float)
                )
                emb_avg = sum_emb / count.unsqueeze(-1)
                output.append(emb_avg)
                output_ids.append(unique_ids)
            else:
                output.append(embs)
                output_ids.append(ids)
        output = pad_sequence(output, batch_first=True, padding_value=0)
        output_ids = pad_sequence(output_ids, batch_first=True, padding_value=pad_val)
        output_pad_mask = output_ids == pad_val
        output_pad_mask.to(embs_.device)

        return output, output_ids, output_pad_mask

    def forward(self, x, mask = None, return_embedding=False):
        # cancer, sex, age, time, event, gene, maf, focal_cna, broad_cna, pad_mask= x   #broad is binary
        cancer, cancer_detailed, sex, age, race, time, event, gene_id, gene_emb, maf, focal_cna_id, focal_cna_alteration, focal_cna_gene_emb, seg_id, seg_start, seg_end, seg_mean = x     #broad is emb

        clin_pad_mask = torch.zeros(cancer.size(0), self.config.n_clin_vars, dtype=torch.bool, device = cancer.device)
        clin_pad_mask[:, 2] = (sex == self.sex_nan_idx).squeeze()
        clin_pad_mask[:, 3] = (torch.isnan(age)).squeeze()
        clin_pad_mask[:, 4] = (race == self.race_nan_idx).squeeze()
        clin_pad_mask[:, 5] = (torch.isnan(time)).squeeze()

        #remove nan
        age = torch.where(torch.isnan(age), torch.zeros_like(age), age)
        time = torch.where(torch.isnan(time), torch.zeros_like(time), time)
        maf_nan_mask = torch.isnan(maf)
        maf = torch.where(maf_nan_mask, torch.zeros_like(maf), maf)

        #clinical
        cancer_emb = self.embed_cancer_type(cancer)
        cancer_d_emb = self.embed_cancer_type_detailed(cancer_detailed)
        sex_emb = self.embed_sex(sex)
        age_emb = self.embed_age(age).unsqueeze(1)
        race_emb = self.embed_race(race)
        surv_emb = self.embed_surv(time, event)

        #gene
        maf_emb = self.embed_maf(maf.unsqueeze(-1)) 
        gene_maf_cat = torch.cat([gene_emb, maf_emb], dim=-1)
        gene_cat_emb = self.embed_gene(gene_maf_cat)

        focal_cna_alt_emb = self.embed_focal_cna(focal_cna_alteration)
        focal_cna_cat = torch.cat([focal_cna_gene_emb, focal_cna_alt_emb], dim=-1)
        focal_cna_emb = self.embed_gene(focal_cna_cat)
        
        gene_avg = torch.cat([gene_cat_emb, focal_cna_emb], dim=1)
        gene_avg_ids = torch.cat([gene_id, focal_cna_id], dim=1)

        gene_avg_emb, gene_avg_emb_ids, gene_pad_mask = self.mean_pool_by_id(gene_avg, gene_avg_ids, self.config.gene_id_vocab_size, pool = self.config.pool_gene)

        # broad_cna_emb = self.embed_broad_cna(broad_cna)                       #broad is binary
        seg = torch.stack([seg_start, seg_end, seg_mean], dim=-1)               #broad is emb
        seg_emb = self.embed_seg(seg)                                           #broad is emb
        seg_avg_emb, seg_avg_emb_ids, seg_pad_mask = self.mean_pool_by_id(seg_emb, seg_id, self.config.seg_id_vocab_size, pool = self.config.pool_seg)

        if mask is not None:
            mask_emb = self.embed_mask(torch.tensor(0, device = cancer_emb.device))
            cancer_emb = torch.where(mask[:,0], mask_emb.expand_as(cancer_emb), cancer_emb)
            cancer_d_emb = torch.where(mask[:,1], mask_emb.expand_as(cancer_d_emb), cancer_d_emb)
            surv_emb = torch.where(mask[:,2], mask_emb.expand_as(surv_emb), surv_emb)
            
        clin_emb = torch.cat([cancer_emb, cancer_d_emb, sex_emb, age_emb, race_emb, surv_emb], dim=1) 
        # tumor_emb = torch.cat([cancer_emb, sex_emb, age_emb, surv_emb, gene_summary_emb, focal_cna_emb, broad_cna_emb], dim=1) #broad is binary
        
        tumor_emb = torch.cat([clin_emb, gene_avg_emb, seg_avg_emb], dim=1)        #broad is emb
        pad_mask = torch.cat([clin_pad_mask, gene_pad_mask, seg_pad_mask], dim=1) #broad is emb
        tumor_emb = tumor_emb.masked_fill(pad_mask.unsqueeze(-1), 0)
        
        clin_pos = self.embed_clin_id(torch.arange(self.config.n_clin_vars, device = cancer_emb.device)).repeat(cancer.size(0), 1, 1)
        gene_pos = self.embed_gene_id(gene_avg_emb_ids)
        seg_pos = self.embed_seg_id(seg_avg_emb_ids)
        pos_emb = torch.cat([clin_pos, gene_pos, seg_pos], dim=1)
        tumor_emb = tumor_emb + pos_emb
        
        for block in self.blocks:
            tumor_emb = block(tumor_emb, pad_mask=pad_mask)  # [B, G, emb_dim]

        tumor_emb = self.norm(tumor_emb)
        
        #set pad tokens emb to zero
        tumor_emb = tumor_emb.masked_fill(pad_mask.unsqueeze(-1), 0)

        outputs = []
        for i,head in enumerate([self.cancer_head, self.cancer_detailed_head, self.sex_head, self.age_head, self.race_head, self.survival_head]):
            outputs.append(head(tumor_emb[:,i,:]))

        if return_embedding:
            return outputs, tumor_emb
        return outputs
    
def create_mask(N, M, probs=None):
    if probs is None:
        probs = torch.full((M,), 0.5)
    else:
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs, dtype=torch.float32)
    rand_tensor = torch.rand(N, M)
    bool_tensor = rand_tensor < probs.unsqueeze(0)
    bool_tensor = bool_tensor.unsqueeze(-1).unsqueeze(-1)
    return bool_tensor

# Assume these loss functions are defined:
classification_loss_fn = nn.CrossEntropyLoss()

def train_masked_loss_with_test(model, config, train_loader, test_loader, optimizer, device, num_epochs=15, saveName=None):
    """
    Training loop with masked loss (for cancer type, detailed type, and survival)
    and evaluation at the end of each epoch.
    """

    best_c_index = -float('inf')
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0
        num_batches = 0

        # --- Training Phase ---
        with tqdm(enumerate(train_loader), total=len(train_loader),desc='TRAINING') as pbar:
            for batch_idx, batch in pbar:
        # for batch in tqdm(train_loader, desc=f"TRAINING Epoch {epoch+1}/{num_epochs}"):
                batch = [item.to(device) for item in batch]
                # Unpack the batch (assumed collate returns 17 items)
                (cancer, cancer_d, sex, age, race, time, event, 
                gene_id, gene_emb, maf, focal_cna_id, focal_cna_alteration, focal_cna_gene_emb, 
                seg_id, seg_start, seg_end, seg_mean) = batch

                # Create mask for the three heads (cancer, cancer_detailed, survival)
                mask = create_mask(cancer.size(0), 3, probs=[0.5, 0.5, 0.5]).to(device)
                optimizer.zero_grad()

                outputs, output_embedding = model(batch, mask=mask, return_embedding = True)
                # print(outputs[2],'\n---\n',output_embedding[2],'\n---\n')
                # Make sure to use the proper indices:
                cancer_logits = outputs[0]          # shape: [B, num_cancer_classes]
                cancer_detailed_logits = outputs[1]   # shape: [B, num_cancer_detailed_classes]
                survival_logits = outputs[5]          # shape: [B, 1]

                # Extract masks for each head
                mask_cancer = mask[:, 0].view(-1)          # shape: [B]
                mask_cancer_detailed = mask[:, 1].view(-1)   # shape: [B]
                mask_survival = mask[:, 2].view(-1)          # shape: [B]

                # Compute losses only over the masked samples:
                if mask_cancer.sum() > 0:
                    loss_cancer = classification_loss_fn(
                        cancer_logits[mask_cancer],
                        cancer[mask_cancer].long().view(-1)
                    )
                else:
                    loss_cancer = torch.tensor(0.0, device=device)

                if mask_cancer_detailed.sum() > 0:
                    loss_cancer_detailed = classification_loss_fn(
                        cancer_detailed_logits[mask_cancer_detailed],
                        cancer_d[mask_cancer_detailed].long().view(-1)
                    )
                else:
                    loss_cancer_detailed = torch.tensor(0.0, device=device)

                # Survival targets: concatenate time and event so that shape becomes [B, 2]
                survival_targets = torch.cat([time, event], dim=1)
                valid_time_mask = ~torch.isnan(time.view(-1))
                mask_survival = mask_survival & valid_time_mask
                if mask_survival.sum() > 0:
                    # Apply the same mask to survival targets and logits.
                    masked_survival = survival_targets[mask_survival]
                    masked_risk = survival_logits[mask_survival].view(-1)  # expecting shape [N]
                    loss_survival = negative_log_partial_likelihood(masked_survival, masked_risk)
                else:
                    loss_survival = torch.tensor(0.0, device=device)

                total_loss = loss_cancer + loss_cancer_detailed + loss_survival
                total_loss.backward()
                optimizer.step()
                pbar.set_postfix({'Epoch':f'{epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}'})

                train_loss_total += total_loss.item()
                num_batches += 1


            avg_train_loss = train_loss_total / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

        # --- Testing / Evaluation Phase ---
        model.eval()
        all_cancer_preds = []
        all_cancer_d_preds = []
        all_cancer_targets = []
        all_cancer_d_targets = []
        all_survival_risks = []
        all_times = []
        all_events = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="TESTING"):
                batch = [item.to(device) for item in batch]
                (cancer, cancer_d, sex, age, race, time, event, 
                 gene_id, gene_emb, maf, focal_cna_id, focal_cna, focal_cna_gene_emb,
                 seg_id, seg_start, seg_end, seg_mean) = batch

                # For evaluation, call forward with mask=None (or a mask that does not alter the outputs)
                mask = create_mask(cancer.size(0), 3, probs=[1, 1, 1]).to(device)
                outputs = model(batch, mask=mask)
                cancer_logits = outputs[0]
                cancer_d_logits = outputs[1]
                survival_logits = outputs[5].squeeze()  # shape: [B] if output is [B,1]

                all_cancer_preds.append(torch.argmax(cancer_logits, dim=1))
                all_cancer_targets.append(cancer.long().squeeze())
                all_cancer_d_preds.append(torch.argmax(cancer_d_logits, dim=1))
                all_cancer_d_targets.append(cancer_d.long().squeeze())
                all_survival_risks.append(survival_logits)
                all_times.append(time)
                all_events.append(event)

        # Concatenate predictions/targets over test set
        all_cancer_preds = torch.cat(all_cancer_preds)
        all_cancer_targets = torch.cat(all_cancer_targets)
        test_accuracy = (all_cancer_preds == all_cancer_targets).float().mean().item()

        all_cancer_d_preds = torch.cat(all_cancer_d_preds)
        all_cancer_d_targets = torch.cat(all_cancer_d_targets)
        test_accuracy_detailed = (all_cancer_d_preds == all_cancer_d_targets).float().mean().item()

        all_survival_risks = torch.cat(all_survival_risks).cpu().numpy() 
        all_times = torch.cat(all_times).cpu().numpy()
        all_events = torch.cat(all_events).cpu().numpy()
        survival_array = np.column_stack((all_times, all_events))
        c_index_value = c_index(all_survival_risks, survival_array)

        print(f"Epoch {epoch+1} - Test Accuracy: {test_accuracy*100:.2f}%, Detailed: {test_accuracy_detailed*100:.2f}%, Survival C-Index: {c_index_value:.4f}")

        if c_index_value > best_c_index:
            best_c_index = c_index_value
            best_epoch = epoch
            if saveName is not None:
                torch.save(model.state_dict(), saveName)
                print(f"Saved best model to {saveName} at epoch {epoch+1}")

    print(f"Best Survival C-Index: {best_c_index:.4f} at epoch {best_epoch+1}")

def train_somatt(modelClass, configClass, dataset, data_df, saveName, n_folds=None, test_size=None,
                 num_epochs=15, lr=1e-4, batch_size = 100, device='cuda:0', collate_fn=None):

    folds = getPatientGroupedLoaders(dataset, data_df, n_folds=n_folds, test_size=test_size,
                                     batch_size=batch_size, collate=collate_fn)

    config = configClass()  # Instantiate your configuration.

    for i, (train_loader, test_loader) in enumerate(folds):
        model = modelClass(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print(f'\nFOLD {i+1}')
        current_saveName = None
        if saveName is not None:
            current_saveName = './best_models/' + saveName.split('.')[0] + f'_fold{i+1}.pt'
        train_masked_loss_with_test(model, config, train_loader, test_loader, optimizer, device, num_epochs=num_epochs, saveName=current_saveName)