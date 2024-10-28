#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from saveAndLoad import *
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
import concurrent.futures
from mutated_sequence_generator import *

def printAllHead(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):
        print(df.head())
def printAll(x):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):
        print(x)


# ### Load all data

# In[ ]:


data = pickleLoad('consolidated_data.pkl')
seqs = pickleLoad('dna_seq_by_hgncId.pkl')


# In[ ]:



    
    # # a list of mutations for every index
    # def getAllMutations(self):
    #     exons_concat_ref = []
    #     exons_concat_mut = []
    #     isos_ref = []
    #     isos_mut = []
    #     canonical_ref = []
    #     canonical_mut = []

    #     getRefAndMutatedSeq = self.getRefAndMutatedSeq

    #     for idx in tqdm(range(len(self.data))):
    #         isoforms_mutated_aa, isoforms_aa, all_exons_mutated_aa, all_exons_aa = getRefAndMutatedSeq(idx)
    #         isos_ref.append(isoforms_aa)
    #         isos_mut.append(isoforms_mutated_aa)
    #         canonical_ref_aa = isoforms_aa['1'] if '1' in isoforms_aa else None
    #         canonical_mut_aa = isoforms_mutated_aa['1'] if '1' in isoforms_mutated_aa else None
    #         canonical_mut.append(canonical_mut_aa)
    #         canonical_ref.append(canonical_ref_aa)
    #         exons_concat_ref.append(all_exons_aa)
    #         exons_concat_mut.append(all_exons_mutated_aa)
    #     return isos_ref,isos_mut,exons_concat_ref,exons_concat_mut, canonical_ref, canonical_mut

seqGen = mutatedSequenceGenerator(data,seqs)
getMutAA_isoforms = lambda x: seqGen.getRefAndMutatedSeq(x)[0]
getRefAA_isoforms = lambda x: seqGen.getRefAndMutatedSeq(x)[1]
getMutAA_concat = lambda x: seqGen.getRefAndMutatedSeq(x)[2]
getRefAA_concat = lambda x: seqGen.getRefAndMutatedSeq(x)[3]


# In[ ]:


# i=0
# seqGen.printInfo(i)
# isoforms_mutated_aa, isoforms_aa, all_exons_mutated_aa, all_exons_aa = seqGen.getRefAndMutatedSeq(i,verbose=True)
# print(isoforms_mutated_aa['1'])
# print(isoforms_aa['1'])


# In[ ]:


[isos_ref,isos_mut,exons_concat_ref,exons_concat_mut, canonical_ref, canonical_mut], tumors = seqGen.getAllMutations()
pickleSave(isos_ref,'../aa/','isos_ref.pkl')
pickleSave(isos_mut,'../aa/','isos_mut.pkl')
pickleSave(exons_concat_ref,'../aa/','exons_concat_ref.pkl')
pickleSave(exons_concat_mut,'../aa/','exons_concat_mut.pkl')
pickleSave(canonical_ref,'../aa/','canonical_ref.pkl')
pickleSave(canonical_mut,'../aa/','canonical_mut.pkl')
pickleSave(tumors,'../aa/','tumors.pkl')


# In[ ]:


## CASE 1

# mutation     --> HIST1H3B (previous hugo symbol of H3C2)              
# mutation_map --> HGNC:4776              
                                        
# HGNC:4776 not found in hgnc_uniprot.
# look up HGNC:4776 on uniprot --> P68431
# look up HGNC:4776 on HGNC --> H3C2, H3 clustered histone 2

# uniprot_hgnc['P68431'] --> HGNC:4774

# look up HGNC:4774 on HGNC --> H3C12, H3 clustered histone 12
# uniprotGtf entry for P68431 contains multiple transcriptIds with _dup1, _dup2
# the ENSEMBL ID of dup2 matches H3C2

# solution: 
# 1. create exons for each transcriptId
# 2. if there are more than one transcriptId, try to resolve
# now these are thrown out, 3148 hgncIds not found in uniprot (0.17%) (probably not all of these are transcriptId issues)

## CASE 2

# mutation      --> PIK3CA
# mutation_map  --> HGNC:8975

# hgnc_uniprot              --> C9J951
# look up C9J951 on HGNC    --> PIK3CA
# look up PIK3CA on uniprot --> P42336

# solution:
# there are multiple uniprot Ids that map to the same HGNC:ID 
# write the hgnc_uniprot algorithm to check if there are multiple Ids and prefer the one in SwissProt 


#### NEED TO CHECK CHROMOSOME ON UNIPROT GTF ####


# In[ ]:


# for ni,i in enumerate(data):
#     if 'Del' in i[4]:
#         a1 = type(i[13])==str and len(i[13])>2 
#         a2 = type(i[14])==str and len(i[14])>2
#         if a1 or a2: assert False, ni
#     # if 'Ins' in i[4]:
#     #     ar = type(i[12])==str and len(i[12])>2
#     #     if ar: assert False, ni


# In[3]:


# from saveAndLoad import *
# isos_mut = pickleLoad('./aa/isos_mut.pkl')


# In[8]:


# asdf = sorted(tuple(list(isos_mut[0].items())))
# dict(asdf) == isos_mut[0]


# In[ ]:




