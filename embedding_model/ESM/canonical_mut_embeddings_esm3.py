import warnings

# Suppress the specific FutureWarning message about `torch.load`
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from saveAndLoad import *
import numpy as np
from tqdm import tqdm
device = 'cuda:0'

def feature_select(sequence_features):
    select_feature = 'cls'

    # Apply feature selection based on select_feature
    if select_feature == 'patch':  # Default behavior for 'patch'|
        sequence_features = [features[1:] for features in sequence_features]  # Skipping the CLS token
    elif select_feature == 'cls_patch':
        sequence_features = sequence_features  # Keep the entire sequence including CLS
    elif select_feature == 'cls':  # Keep only the CLS token
        sequence_features = [features[:1] for features in sequence_features]
    else:
        raise ValueError(f'Unexpected select feature: {select_feature}')    
    return sequence_features

def encode_esm3(sequences,device = device):
    device = torch.device(device) if type(device) == str else device

    client = ESM3.from_pretrained("esm3_sm_open_v1", device=device)

    protein_objects = [ESMProtein(sequence=seq) for seq in sequences]
    
    def get_features(obj):
        protein_tensor = client.encode(obj)
        output = client.forward_and_sample(
            protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
        return output.per_residue_embedding
    
    sequence_features = [get_features(obj) for obj in protein_objects]
    sequence_features = feature_select(sequence_features)
    
    return sequence_features

# muts = pickleLoad('../../aa/canonical_mut.pkl')
# mut2ref = pickleLoad('../../aa/idxMap_canonical_mut_to_ref.pkl')
# mut2type = pickleLoad('../../aa/mut2type.pkl')
# refs = pickleLoad('../../aa/canonical_ref.pkl')
# data = pickleLoad('../../data_processing/consolidated_data.pkl')
# tumors = pickleLoad('../../aa/tumors.pkl') 

# mut_seqs = []
# ref_seqs = []

# #to filter out nonstop mutations 
# for seq,typ in zip(muts,mut2type): 
#     if seq is None or typ == 'Nonstop_Mutation': 
#         mut_seqs.append('')
#         continue
#     seq = seq.split('*')[0]
#     mut_seqs.append(seq)

# #to filter out mutations that are matched to bad references
# for seq in refs: 
#     if seq is None or '*' in seq: 
#         ref_seqs.append('')
#         continue
#     ref_seqs.append(seq)

# #filter
# all_seqs = []
# for i,mut in tqdm(enumerate(mut_seqs), total = len(mut_seqs)):
#     ref = ref_seqs[mut2ref[i]]
#     typ = mut2type[i]
#     if mut == '' or ref == '' or mut == ref: 
#         all_seqs.append('')
#         continue
#     all_seqs.append(mut)

# #save the list of all_seqs to file as line separated strings
# with open('/data/dandreas/SomaticMutationsLLM/aa/canonical_mut_seqs.txt','w') as f: 
#     for seq in all_seqs: 
#         f.write(seq+'\n')

#load the list of all_seqs from file as line separated strings
with open('/data/dandreas/SomaticMutationsLLM/aa/canonical_mut_seqs.txt','r') as f:
    all_seqs = [line.strip() for line in f]

try:
    mut_embeddings = np.load('/data/dandreas/SomaticMutationsLLM/aa/canonical_mut_cls_embeddings_esm3.npy')
except:
    mut_embeddings = np.zeros((len(all_seqs),1536),dtype=np.float32)
print(f'getting {len(mut_embeddings)} embeddings')
load_val = lambda x: torch.tensor([x]).to(device)

# count = 0
# for i,seq in tqdm(enumerate(all_seqs), total = len(all_seqs)):
#     if seq == '': continue
#     if mut_embeddings[i].sum() != 0: continue
#     try:
#         with torch.no_grad():
#             embedding = encode_esm3([seq],device = device)[0].detach().to('cpu').numpy()
#             mut_embeddings[i] = embedding
#             if i % 100 == 0:
#                 np.save('/data/dandreas/SomaticMutationsLLM/aa/canonical_mut_cls_embeddings_esm3.npy',mut_embeddings)
#     except:
#         count += 1
#         print(i,'failed')
# print(count,'failed')

batch_size = 30
#load the list of failed batches
try:
    with open('./temp/failed_batches_canonical_mut_embeddings_esm3.txt','r') as f:
        failed_batches = [int(line.strip()) for line in f]
except:
    failed_batches = []


for start_idx in tqdm(range(0, len(all_seqs), batch_size), total=len(all_seqs)//batch_size+1):
    if start_idx in failed_batches: continue
    # Get the batch slice
    end_idx = min(start_idx + batch_size, len(all_seqs))
    batch_indices = []
    batch_seqs = []

    batch_filled_or_empty = True
    for i in range(start_idx, end_idx):
        if all_seqs[i] != '' and mut_embeddings[i].sum() == 0:
            batch_filled_or_empty = False
            break

    if batch_filled_or_empty:
        # Every sequence in this batch is either empty or already has embeddings
        continue

    # Collect valid sequences and corresponding indices
    for i in range(start_idx, end_idx):
        seq = all_seqs[i]
        # Skip if sequence is empty
        if seq == '':
            continue
        # Skip if embedding is already set (non-zero sum)
        if mut_embeddings[i].sum() != 0:
            continue

        batch_indices.append(i)
        batch_seqs.append(seq)

    # If there are no valid sequences in this batch, just continue
    if len(batch_seqs) == 0:
        continue

    # Encode the valid sequences in a single batch call
    try:
        with torch.no_grad():
            # Encode all valid seqs in the batch
            batch_embeddings = encode_esm3(batch_seqs, device=device)

        # Assign embeddings back to the correct positions in mut_embeddings
        for idx_in_batch, i in enumerate(batch_indices):
            mut_embeddings[i] = batch_embeddings[idx_in_batch].detach().to('cpu').numpy()

        np.save('/data/dandreas/SomaticMutationsLLM/aa/canonical_mut_cls_embeddings_esm3.npy',
                mut_embeddings)

    except Exception as e:
        print(f"Batch starting at {start_idx} failed with error: {e}")
        failed_batches.append(start_idx)
        with open('./temp/failed_batches_canonical_mut_embeddings_esm3.txt','a') as f:
            f.write(str(start_idx)+'\n')
        exit(1)

# Finally, save the embeddings and print the number of failures
np.save('/data/dandreas/SomaticMutationsLLM/aa/canonical_mut_cls_embeddings_esm3.npy', mut_embeddings)
