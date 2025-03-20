from training import *

# mut_embeddings = np.load('/data/dandreas/SomaticMutationsLLM/aa/canonical_mut_cls_embeddings_esm3.npy')
# ref_embeddings = np.load('/data/dandreas/SomaticMutationsLLM/aa/canonical_ref_cls_embeddings_esm3.npy')

mut_embeddings = np.load('../aa/canonical_mut_norm_embeddings_esm2.npy')
ref_embeddings = np.load('../aa/canonical_ref_embeddings_esm2.npy')

# mut_embeddings = np.load('../aa/canonical_mut_norm_embeddings_esmC.npy')
# ref_embeddings = np.load('../aa/canonical_ref_cls_embeddings_esmC.npy')

tumors = pickleLoad('../aa/tumors.pkl')
assays = pickleLoad('../aa/assays.pkl')
device = 'cuda:1'

#CANCER TYPE
data_df_emb, saveName = get_df('data_1_00percentMinCancerType.csv', suffix = 'MSK-IMPACT468')
data_df_emb = data_df_emb[data_df_emb['assay']=='MSK-IMPACT468']
dataset = Dataset_Assay(data_df_emb, 'CANCER_TYPE_INT', mut_embeddings, ref_embeddings, tumors, assays, device)
# dataset = Dataset_Classification_MutationsOnly(data_df_emb, 'CANCER_TYPE_INT', mut_embeddings, tumors, device)
n_labels = max(data_df_emb['CANCER_TYPE_INT'].unique())+1
print('n labels:',n_labels)
train_cancer_type(Classifier, Config_Att, dataset, data_df_emb, saveName, n_labels, test_size = .2, device = device)

#SURVIVAL
data_df_emb, saveName = get_df('data_1_00_survival_ratio1.csv', suffix = 'MSK-IMPACT468')
dataset = Dataset_Assay_Survival(data_df_emb, mut_embeddings, ref_embeddings, tumors, assays, device)
# dataset = Dataset_Survival_MutationsOnly(data_df_emb, mut_embeddings, tumors, device) #gives assert false
train_survival(Classifier, dataset, data_df_emb, saveName, test_size = .2, device = device)
