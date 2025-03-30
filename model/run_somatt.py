from models import *
from somatt import *
from saveAndLoad import pickleLoad

tumors = pickleLoad('../aa/tumors.pkl')
map_tumorBarcode_to_clinicalSampleIdx = pickleLoad('../data_processing/map_tumorBarcode_to_clinicalSampleIdx.pkl')
mut_embeddings = np.load('../aa/canonical_mut_norm_embeddings_esm2.npy')
ref_embeddings = np.load('../aa/canonical_ref_embeddings_esm2.npy')
assays = pickleLoad('../aa/assays.pkl')
device = 'cuda:0'
sex_label_mapping = pickleLoad('../labeled_data/label_mappings/label_mapping_SEX_somatt_data_df.pkl')
race_label_mapping = pickleLoad('../labeled_data/label_mappings/label_mapping_RACE_somatt_data_df.pkl')
arm_label_mapping = pickleLoad('../labeled_data/label_mappings/label_mapping_ARM_somatt_data_df.pkl')
cancer_type_label_mapping = pickleLoad('../labeled_data/label_mappings/label_mapping_CANCER_TYPE_somatt_data_df.pkl')
cancer_type_detailed_label_mapping = pickleLoad('../labeled_data/label_mappings/label_mapping_CANCER_TYPE_DETAILED_somatt_data_df.pkl')
n_cancer_types = len(cancer_type_label_mapping)
n_cancer_type_d = len(cancer_type_detailed_label_mapping)
n_seg_id = len(arm_label_mapping)
sex_nan_idx = len(sex_label_mapping)-1
assert math.isnan(sex_label_mapping[sex_nan_idx])
race_nan_idx = len(race_label_mapping)-1
assert math.isnan(race_label_mapping[race_nan_idx])

class Config_Somatt:
    n_layer: int = 3
    emb_dim: int = 640 #1152 esmC #1536 esm3 #640 esm2
    input_dim: int = 640
    dropout: float = 0.0
    bias: bool = False
    gene_id_vocab_size : int = 1433
    cancer_type_vocab_size: int = n_cancer_types
    cancer_type_detailed_vocab_size: int = n_cancer_type_d
    norm_fn: nn.Module = nn.LayerNorm
    position_embedding: bool = False
    sex_nan_idx: int = sex_nan_idx
    sex_vocab_size: int =  len(sex_label_mapping)
    race_nan_idx: int = race_nan_idx
    race_vocab_size: int = len(race_label_mapping)
    num_heads: int = 1
    lin_proj: nn.Module = LinProj
    rbf_params = (0,1,16)
    n_clin_vars: int = 6
    maf_emb_dim: int = 16
    seg_id_vocab_size: int = n_seg_id
    pool_gene: bool = False
    pool_seg: bool = True
    broad_cna_vocab_size: int = 2   
    focal_cna_vocab_size: int = 2   
    event_vocab_size: int = 2
    embed_surv: nn.Module = EmbedSurv_rbf_2heads

# Example usage:
config = Config_Somatt()  # Your defined config class for Somatt.
model = Somatt(config)
collate_somatt_with_config = partial(collate_somatt, config=config)

data_df = pd.read_pickle('../labeled_data/somatt_data_df.pkl')
label_counts = data_df['CANCER_TYPE'].value_counts().to_dict()
min_cancer_type = .01
filter_rarity = lambda x: label_counts[x]>=min_cancer_type
min_cancer_type = int(len(data_df)*min_cancer_type)
data_df = data_df[data_df['CANCER_TYPE'].apply(filter_rarity)]

ds = Dataset_Somatt(data_df, mut_embeddings, ref_embeddings, tumors, assays, device)

train_somatt(Somatt, Config_Somatt, ds, data_df, batch_size = 100, saveName=None, test_size=0.2, num_epochs=15, lr=1e-4, device=device, collate_fn = collate_somatt_with_config)
