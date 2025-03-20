#!/usr/bin/env python

import argparse
import warnings
import sys
import torch
import numpy as np
from tqdm import tqdm
import os

# ESM imports
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig, LogitsConfig
from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm3C_util import encode_esm3, encode_esmC

# For unpickling reference sequences
from saveAndLoad import pickleLoad  # adjust if needed or inline your own load logic

#############################################
# Suppress Torch Load FutureWarning
#############################################
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

# Set your device
device = 'cuda:0'  # or whichever GPU device you have available

# Define which amino acids to mutate to:
# Option 1: Hard-code the 20 standard amino acids
# Option 2: Dynamically gather from your reference set
ALL_AMINO_ACIDS = [
    'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G',
    'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
    'T', 'W', 'Y', 'V'
]

##############################################################################
# Main script
##############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Generate random SNPs for each reference sequence and embed them using ESM3, storing a 3D array."
    )
    parser.add_argument(
        "--seqs-path",
        required=True,
        type=str,
        help="Path to the pickle file containing reference sequences (list or similar)."
    )
    parser.add_argument(
        "--emb-path",
        required=True,
        type=str,
        help="Path to the .npy file for loading/saving the final 3D embeddings array."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for ESM3 embedding (default=1024)."
    )

    parser.add_argument(
        "--esm-model",
        type=str,
        required = True,
        help="esm3 or esmC"
    )

    parser.add_argument(
        "--n-mutations",
        type=int,
        default=33,
        help="Number of random SNP (amino-acid) mutations to generate per sequence (default=33)."
    )
    args = parser.parse_args()

    seqs_path = args.seqs_path
    emb_path = args.emb_path
    batch_size = args.batch_size
    esm_model = args.esm_model
    n_mutations = args.n_mutations
    
    if esm_model == 'esm3': encode = lambda seqs: encode_esm3(seqs, device)
    elif esm_model == 'esmC': encode = lambda seqs: encode_esmC(seqs, device)
    else: raise ValueError(f"Unexpected --esm-model: {esm_model}, expected esm3 or esmC")

    # 1) Load reference sequences
    with open(seqs_path, 'r') as f:
        ref_seqs = [line.strip() for line in f]

    # 2) Prepare a 3D array to hold embeddings:
    #    Shape: (num_ref_seqs, n_mutations, emb_dim)
    num_ref = len(ref_seqs)
    emb_dim = 1536 if esm_model == 'esm3' else 1152  # ESM3 CLS dimension

    # Attempt to load existing embeddings if file exists
    if os.path.isfile(emb_path):
        mut_embeddings_3d = np.load(emb_path)
        if mut_embeddings_3d.shape != (num_ref, n_mutations, emb_dim):
            raise ValueError(
                f"Existing embedding file shape {mut_embeddings_3d.shape} does not match "
                f"expected shape {(num_ref, n_mutations, emb_dim)}"
            )
        print(f"Loaded existing embeddings from {emb_path}: shape = {mut_embeddings_3d.shape}")
    else:
        mut_embeddings_3d = np.zeros((num_ref, n_mutations, emb_dim), dtype=np.float32)
        print(f"Created new embeddings array with shape {mut_embeddings_3d.shape}")

    # 3) For each reference sequence, generate the 33 random mutated sequences
    #    along with the (ref_idx, mut_idx) positions in the final array.
    #    We will flatten them into a single list for sorting/embedding in batch.
    all_mutated_seqs = []
    index_map = []  # will store (ref_idx, mut_idx)

    # If you prefer to sample from all AA present in the entire dataset, you can gather them first:
    # Alternatively, we just use the standard 20 amino acids above: ALL_AMINO_ACIDS

    # Build a single big list of mutated sequences
    for i, seq in enumerate(ref_seqs):
        if not seq:  # skip empty
            continue
        
        seq_len = len(seq)
        # Choose random positions (no replacement)
        positions = np.random.choice(seq_len, size=n_mutations, replace=False)
        
        for j, pos in enumerate(positions):
            original_aa = seq[pos]
            # Mutate to a different AA
            # keep sampling until we get a different one
            new_aa = original_aa
            while new_aa == original_aa:
                new_aa = np.random.choice(ALL_AMINO_ACIDS)
            
            mut_seq = seq[:pos] + new_aa + seq[pos+1:]
            all_mutated_seqs.append(mut_seq)
            index_map.append((i, j))

    # 4) We want to see which entries in mut_embeddings_3d are not yet embedded
    #    (sum == 0 means presumably all zeros)
    #    We'll store that in a list of (global_idx, length)
    #    where global_idx is the index into all_mutated_seqs/index_map
    need_indices = []
    for global_idx, (i, j) in enumerate(index_map):
        if mut_embeddings_3d[i, j].sum() == 0:
            seq_len = len(all_mutated_seqs[global_idx])
            need_indices.append((global_idx, seq_len))

    if not need_indices:
        print("No mutations need embedding. Exiting.")
        sys.exit(0)

    print(f"Total mutated sequences requiring embedding: {len(need_indices)}")

    # 5) Sort the needed indices by the length of their mutated sequences
    need_indices.sort(key=lambda x: x[1])  # sorts by length ascending

    print(f"Using batch size = {batch_size}")

    # 6) Embed in batches
    num_processed = 0
    try:
        # We only need the global_idx after sorting
        sorted_indices = [ni[0] for ni in need_indices]

        for start_idx in tqdm(range(0, len(sorted_indices), batch_size),
                              total=(len(sorted_indices) // batch_size + 1)):
            end_idx = min(start_idx + batch_size, len(sorted_indices))
            batch_indices = sorted_indices[start_idx:end_idx]

            # Gather the actual sequences for this batch
            batch_seqs = [all_mutated_seqs[gidx] for gidx in batch_indices]
            
            # Embed
            with torch.no_grad():
                batch_features = encode(batch_seqs)
                # batch_features is a list of Tensors, each shape [1, 1536] (if "cls")

            # Save each embedding back in the correct position of mut_embeddings_3d
            for local_i, global_i in enumerate(batch_indices):
                ref_i, mut_i = index_map[global_i]
                # Each batch_features[local_i] is shape (1, 1536)
                emb_vector = batch_features[local_i].detach().cpu().numpy().squeeze(0)  # shape (1536,)
                mut_embeddings_3d[ref_i, mut_i] = emb_vector

            # Save partial progress after each batch in the 3D shape
            np.save(emb_path, mut_embeddings_3d)
            num_processed += len(batch_indices)
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Exception while embedding: {e}")
        # Clear GPU memory
        torch.cuda.empty_cache()
        print(f"Remaining to embed: {len(need_indices) - num_processed}")
        sys.exit(1)

    # 7) Final Save and Exit
    np.save(emb_path, mut_embeddings_3d)
    print(f"Successfully finished all embeddings. Final shape: {mut_embeddings_3d.shape}")
    sys.exit(0)


if __name__ == "__main__":
    main()
