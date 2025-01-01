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

device = 'cuda:1'
checkpoint = 'facebook/esm2_t30_150M_UR50D'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
model.to(device)
model.eval()

mut = pickleLoad('../../aa/canonical_mut.pkl')

mut_seqs = []

for seq in mut: 
    if seq is None: 
        mut_seqs.append('')
        continue
    seq = seq.split('*')[0]
    mut_seqs.append(seq)

mut_embeddings = np.empty((len(mut),640),dtype=np.float32)
print('getting embeddings')
load_val = lambda x: torch.tensor([x]).to(device)

for i,seq in tqdm(enumerate(mut_seqs), total = len(mut)):
    if seq == '': continue
    with torch.no_grad():
        inputs = tokenizer(seq)
        ids = load_val(inputs['input_ids'])
        att = load_val(inputs['attention_mask'])
        output = model(ids, attention_mask=att)
        embedding = output['last_hidden_state'][0][0].detach().to('cpu').numpy()
        mut_embeddings[i] = embedding

np.save('/data/dandreas/SomaticMutationsLLM/aa/canonical_mut_embeddings_esm2.npy',mut_embeddings)