import argparse
import warnings
import sys
import torch
import numpy as np
from tqdm import tqdm

# ESM imports
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig, LogitsConfig
from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from saveAndLoad import pickleLoad  # Adjust if needed or remove if unused
from esm3C_util import encode_esm3, encode_esmC

#############################################
# Suppress Torch Load FutureWarning
#############################################
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

device = 'cuda:0'  # or whichever GPU device you need

def main():
    parser = argparse.ArgumentParser(
        description="Embed amino acid sequences using ESM3, storing results in a NumPy array."
    )
    
    parser.add_argument(
        "--seqs-path",
        required=True,
        type=str,
        help="Path to a text file with one sequence per line."
    )
    parser.add_argument(
        "--emb-path",
        required=True,
        type=str,
        help="Path to the .npy file for loading/saving embeddings."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for embedding (default=1000)"
    )

    parser.add_argument(
        "--esm-model",
        type=str,
        required = True,
        help="esm3 or esmC"
    )
    
    args = parser.parse_args()
    seqs_path = args.seqs_path
    embeddings_path = args.emb_path
    batch_size = args.batch_size
    esm_model = args.esm_model

    if esm_model == 'esm3': encode = lambda seqs: encode_esm3(seqs, device)
    elif esm_model == 'esmC': encode = lambda seqs: encode_esmC(seqs, device)
    else: raise ValueError(f"Unexpected esm model: {esm_model}, expected esm3 or esmC")

    # ----------------------------------------------------
    # 1. Load sequences from file
    # ----------------------------------------------------
    with open(seqs_path, 'r') as f:
        all_seqs = [line.strip() for line in f]

    # ----------------------------------------------------
    # 2. Load or create embeddings
    # ----------------------------------------------------
    try:
        mut_embeddings = np.load(embeddings_path)
        print(f"Loaded existing embeddings from {embeddings_path}: shape = {mut_embeddings.shape}")
        
        # (Optional) Check if the loaded array length matches all_seqs length
        if len(all_seqs) != mut_embeddings.shape[0]:
            raise ValueError(
                f"Embeddings file length mismatch: {mut_embeddings.shape[0]} vs. {len(all_seqs)} sequences."
            )
    except FileNotFoundError:
        # Create new embeddings array if file doesn't exist
        if esm_model == 'esm3': dim = 1536 
        elif esm_model == 'esmC': dim = 1152
        else: raise ValueError(f"Unexpected esm model: {esm_model}, expected esm3 or esmC")

        mut_embeddings = np.zeros((len(all_seqs), dim), dtype=np.float32)
        print(f"Created new embeddings array with shape {mut_embeddings.shape}")

    # ----------------------------------------------------
    # 3. Build list of indices that still need embedding
    #    (non-empty seq, sum=0)
    # ----------------------------------------------------
    need_indices = []
    for i, seq in enumerate(all_seqs):
        if seq != '' and mut_embeddings[i].sum() == 0:
            need_indices.append(i)

    if not need_indices:
        print("No sequences need embedding. Exiting.")
        sys.exit(0)

    # ----------------------------------------------------
    # 4. Sort these indices by sequence length (ascending)
    # ----------------------------------------------------
    need_indices.sort(key=lambda i: len(all_seqs[i]))
    
    print(f"Total sequences needing embedding: {len(need_indices)}")
    print(f"Using batch size = {batch_size}")

    # ----------------------------------------------------
    # 5. Embed in batches
    # ----------------------------------------------------
    num_processed = 0
    try:
        for start_idx in tqdm(
            range(0, len(need_indices), batch_size),
            total=(len(need_indices) // batch_size + 1)
        ):
            end_idx = min(start_idx + batch_size, len(need_indices))
            batch_indices = need_indices[start_idx:end_idx]
            batch_seqs = [all_seqs[idx] for idx in batch_indices]

            with torch.no_grad():
                batch_features = encode(batch_seqs)

            # Save each embedding back in the correct row
            for local_i, global_i in enumerate(batch_indices):
                mut_embeddings[global_i] = batch_features[local_i].detach().cpu().numpy()

            # Save partial progress after each batch
            np.save(embeddings_path, mut_embeddings)
            num_processed += len(batch_indices)
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Exception while embedding: {e}")
        # Clear GPU memory
        torch.cuda.empty_cache()
        # Return non-zero exit code => signals failure to external process
        print('remaining:',len(need_indices)-num_processed)
        sys.exit(1)

    # ----------------------------------------------------
    # 6. Final Save and Exit
    # ----------------------------------------------------
    np.save(embeddings_path, mut_embeddings)
    print("Successfully finished all embeddings.")
    sys.exit(0)

if __name__ == "__main__":
    main()
