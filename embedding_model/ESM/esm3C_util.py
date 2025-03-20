from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig, LogitsConfig
from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
import torch

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

def encode_esm3(sequences, device):
    """
    Encode a list of amino acid sequences with ESM3,
    returning a list of torch tensors (one per sequence).
    """
    device = torch.device(device) if isinstance(device, str) else device
    client = ESM3.from_pretrained('esm3_sm_open_v1', device=device)

    protein_objects = [ESMProtein(sequence=seq) for seq in sequences]

    def get_features(obj):
        protein_tensor = client.encode(obj)
        output = client.forward_and_sample(
            protein_tensor,
            SamplingConfig(return_per_residue_embeddings=True)
        )
        return output.per_residue_embedding
    
    sequence_features = [get_features(obj) for obj in protein_objects]
    return feature_select(sequence_features)

def encode_esmC(sequences, device):
    """
    Encode a list of amino acid sequences with ESM3,
    returning a list of torch tensors (one per sequence).
    """
    device = torch.device(device) if isinstance(device, str) else device
    client = ESMC.from_pretrained('esmc_600m', device=device)

    protein_objects = [ESMProtein(sequence=seq) for seq in sequences]

    def get_features(obj):
        protein_tensor = client.encode(obj)
        output = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings = True)
        )
        return output.embeddings[0]
    
    sequence_features = [get_features(obj) for obj in protein_objects]
    return feature_select(sequence_features)