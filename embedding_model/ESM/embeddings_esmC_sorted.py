import argparse
import warnings
import sys
import torch
import numpy as np
from tqdm import tqdm

# ESM imports
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig, LogitsConfig
from esm.models.esmc import ESMC
from saveAndLoad import pickleLoad  # Adjust if needed or remove if unused

#############################################
# Suppress Torch Load FutureWarning
#############################################
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

device = 'cuda:0'  # or whichever GPU device you need

def feature_select(sequence_features):
    """
    Select which portion(s) of the ESM3 embedding to keep.
    Default is to keep only the [CLS] token (shape: [1, 1536]).
    """
    select_feature = 'cls'
    if select_feature == 'patch':
        # Skip the CLS token, keep the rest
        sequence_features = [features[1:] for features in sequence_features]
    elif select_feature == 'cls_patch':
        # Keep the entire sequence including CLS
        pass
    elif select_feature == 'cls':
        # Keep only the CLS token
        sequence_features = [features[:1] for features in sequence_features]
    else:
        raise ValueError(f'Unexpected select feature: {select_feature}')
    return sequence_features


def encode_esm3(sequences, device=device):
    """
    Encode a list of amino acid sequences with ESM3,
    returning a list of torch tensors (one per sequence).
    """
    device = torch.device(device) if isinstance(device, str) else device
    client = ESMC.from_pretrained("esmc_600m", device=device)

    protein_objects = [ESMProtein(sequence=seq) for seq in sequences]

    def get_features(obj):
        protein_tensor = client.encode(obj)
        output = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )
        return output.embeddings[0]
    
    sequence_features = [get_features(obj) for obj in protein_objects]
    return feature_select(sequence_features)

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
    
    args = parser.parse_args()
    seqs_path = args.seqs_path
    embeddings_path = args.emb_path
    batch_size = args.batch_size

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
        mut_embeddings = np.zeros((len(all_seqs), 1152), dtype=np.float32)
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
                batch_features = encode_esm3(batch_seqs, device=device)

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
