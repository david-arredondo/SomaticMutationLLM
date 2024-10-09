import pandas as pd
import os
# import wandb
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
from transformers import (
    AutoModel,
    AutoTokenizer,
    # DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from datasets import Dataset
from accelerate import Accelerator
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import pickle
from saveAndLoad import *
from tqdm import tqdm

muts = pickleLoad('../../aa/canonical_mut.pkl')
refs = pickleLoad('../../aa/canonical_ref.pkl')
data = pickleLoad('../../data_processing/consolidated_data.pkl')
tumors = pickleLoad('../../aa/tumors.pkl') 
        #0:concat_ref
        #1:concat_mut
        #2:iso_ref
        #3:iso_mut
        #4:canonical_ref [115] is None
        #5:canonical_mut [39] is None
        #6:data
mut2ref = pickleLoad('../../aa/idxMap_canonical_mut_to_ref.pkl')

def diff_indices(shorter, longer):
    # Ensure we treat the longer string as 'mutated' for uniform handling of insertions/deletions
    if shorter==longer: return None

    # Find the starting index where the two strings differ
    start = next((i for i, (o, m) in enumerate(zip(shorter, longer)) if o != m), len(shorter))

    # Calculate the length difference 
    len_diff = len(longer) - len(shorter)

    if len_diff == 0:
        # Find the end index of the replacement by comparing substrings starting from 'start'
        end = start + next((i for i, (o, m) in enumerate(zip(shorter[start:], longer[start:])) if o == m), len(shorter) - start)
        return start, end

    # Return the range of indices as 'insertion' or 'deletion'
    return start, start + len_diff

device = 'cuda:1'
checkpoint = 'facebook/esm2_t30_150M_UR50D'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
model.to(device)
model.eval()

mut_seqs = []
ref_seqs = []

for seq in muts: 
    if seq is None: 
        mut_seqs.append('')
        continue
    seq = seq.split('*')[0]
    mut_seqs.append(seq)

for seq in refs: 
    if seq is None: 
        ref_seqs.append('')
        continue
    seq = seq.split('*')[0]
    ref_seqs.append(seq)

mut_embeddings = np.empty((len(muts),640),dtype=np.float32)
print('getting embeddings')
load_val = lambda x: torch.tensor([x]).to(device)

all_seqs = []
for i,mut in tqdm(enumerate(mut_seqs), total = len(mut_seqs)):
    ref = ref_seqs[mut2ref[i]]
    if mut == '' or ref == '': 
        all_seqs.append(('',0,0,False))
        continue
    if mut == ref: 
        all_seqs.append(('',0,0,False))
        continue
    deletion = len(ref) > len(mut)
    seq = ref if deletion else mut
    (shorter,longer) = (mut,ref) if deletion else (ref,mut)
    start,end = diff_indices(shorter,longer)
    all_seqs.append((seq,start,end,deletion))
all_seqs

count = 0
for i,(seq,start,end,deletion) in tqdm(enumerate(all_seqs), total = len(all_seqs)):
    if seq=='':continue
    if all(mut_embeddings[i]==0): #if not already computed
        try:
            mut_len = end-start
            with torch.no_grad():
                inputs = tokenizer(seq)
                ids = load_val(inputs['input_ids'])
                att = load_val(inputs['attention_mask'])
                output = model(ids, attention_mask=att)
                embedding = output['last_hidden_state'][0][start:end].detach().to('cpu').numpy()
                if mut_len>1: embedding = np.mean(embedding, axis = 0, keepdims = True)
                if deletion: embedding = -embedding
                mut_embeddings[i] = embedding
            if i%1000==0:
                np.save('/data/dandreas/SomaticMutationsLLM/aa/canonical_mut_average_embeddings_esm2.npy',mut_embeddings)
        except:
            count+=1
            print(i,count)

