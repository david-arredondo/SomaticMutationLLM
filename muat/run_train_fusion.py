from train_setup import *

data_snv150 = lambda x: get_snv150_df(x, data, tumors, seqs, ref_aa, mut_aa, return_somatt_df=True)
data_snv150_pos = lambda x: get_snv150_withPos_df(x, data, tumors, seqs, ref_aa, mut_aa, return_somatt_df=True)

# snv
df_class = [
    data_snv150(class_data), 
    data_snv150_pos(class_data)
]

df_class_msk = [
    data_snv150(class_data_msk468),
    data_snv150_pos(class_data_msk468)
] 

df_surv = [
    data_snv150(survival_data),
    data_snv150_pos(survival_data)
]

df_surv_msk = [
    data_snv150(survival_data_msk468),
    data_snv150_pos(survival_data_msk468)
]

def get_fusion_dim(d,pos=False):
    if not pos: return 150
    all_genes = list(get_gene_indexes(d, tumors, data, seqs, ref_aa, mut_aa).keys())
    return 150 + len(all_genes)


def run(dfs, d, surv, e, names, lr = None):
    pos = False
    for c,n in zip(dfs,names):
        print(n)
        fusion_dim = get_fusion_dim(d, pos = pos)
        run_fusion(
            d, 
            c,
            mut_embeddings,
            ref_embeddings,
            tumors,
            assays,
            fusion_dim=fusion_dim,
            survival=surv,
            num_epochs=e,
            saveName = n,
            lr = lr
            )
        pos = not pos

run(df_class, class_data, False, lr = .001, e = 15, names = ['snv150_class_fusion','snv150_pos_class_fusion'])
run(df_class_msk, class_data_msk468, False, lr = .001, e = 15, names = ['snv150_class_fusion_msk468','snv150_pos_class_fusion_msk468'])   
run(df_surv, survival_data, True, lr = .0001, e = 5, names = ['snv150_survival_fusion','snv150_pos_survival_fusion'])
run(df_surv_msk, survival_data_msk468, True, lr = .0001, e = 5, names = ['snv150_survival_fusion_msk468','snv150_pos_survival_fusion_msk468'])