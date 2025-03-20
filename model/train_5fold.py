from training import *

mut_embeddings = np.load('../aa/canonical_mut_norm_embeddings_esm2.npy')
ref_embeddings = np.load('../aa/canonical_ref_embeddings_esm2.npy')
tumors = pickleLoad('../aa/tumors.pkl')
assays = pickleLoad('../aa/assays.pkl')
device = 'cuda:0'

#CANCER TYPE
data_df_emb, saveName = get_df('data_1_00percentMinCancerType.csv')
dataset = Dataset_Assay(data_df_emb, 'CANCER_TYPE_INT', mut_embeddings, ref_embeddings, tumors, assays, device)
n_labels = len(data_df_emb['CANCER_TYPE_INT'].unique())
print('n labels:',n_labels)
train_cancer_type(dataset, data_df_emb, saveName, n_labels, n_folds = 5, device = device)

#SURVIVAL
data_df_emb,saveName = get_df('data_0_00_survival_ratio1.csv')
dataset = Dataset_Assay_Survival(data_df_emb, mut_embeddings, ref_embeddings, tumors, assays, device)
train_survival(dataset, data_df_emb, saveName, n_folds = 5, device = device)