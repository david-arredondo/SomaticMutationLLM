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


class orderedSet:
    def __init__(self,item_is_dict=False):
        self.ordered_set = dict()
        self.counter = 0
        self.item_is_dict = item_is_dict
    
    def add_to_set(self,item):
        if self.item_is_dict:
            item = tuple(sorted(item.items()))
        if item not in self.ordered_set:
            self.ordered_set[item] = self.counter
            self.counter+=1
            return self.counter -1
        return self.get_index(item)
    
    def get_index(self,item):
        if type(item)==dict:
            item = tuple(sorted(item.items()))
        return self.ordered_set.get(item,None)
    
    def return_ordered_list(self):
        if self.item_is_dict:
            return [dict(i) for i in list(self.ordered_set.keys())]
        return list(self.ordered_set.keys())

class mutatedSequenceGenerator:
    def __init__(self,data,seqs):
        self.data = data
        self.seqs = seqs
    
    def printInfo(self,idx):
        start,end,chrom,build,variant_class,isoforms,hgncId,gene_start,gene_end,ref_allele,strand,mutation_value,barcode = self.data[idx]
        print('variant class:',variant_class)
        print('mutation start:',start)
        print('mutation end:',end)
        print('chromosome',chrom)
        print('build',build)
        print('hgnc id:',hgncId)
        print('gene start:',gene_start)
        print('gene end:',gene_end)
        print('isoforms:',isoforms)
        print('reference allele:',ref_allele)
        print('strand:',strand)
        print('mutation:',mutation_value)
        print('tumor sample barcode:',barcode)
    
    def rc(self,s): #reverse complement
        reverse_complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A','N':'N'}
        return ''.join([reverse_complement[i] for i in s[::-1]])
    
    def mutate(self,mutation,mut_start_zero,mut_end_zero,exon_start_zero,exon_end_zero,variant_class,ref_gene,ref_allele):
        
        if 'Ins' in variant_class:
            mutated_gene = ref_gene[:mut_start_zero-1] + mutation + ref_gene[mut_start_zero-1:]
            mutated_exon = mutated_gene[exon_start_zero-1:exon_end_zero + len(mutation)]

        elif 'Del' in variant_class:
            mutated_gene = ref_gene[:mut_start_zero-1] + mutation + ref_gene[mut_end_zero:]
            mutated_exon = mutated_gene[exon_start_zero-1:exon_end_zero - len(ref_allele)]

        else:
            mutated_gene = ref_gene[:mut_start_zero-1] + mutation + ref_gene[mut_end_zero:]
            mutated_exon = mutated_gene[exon_start_zero-1:exon_end_zero]

        return mutated_exon

    def getRefAndMutatedSeq(self,idx,verbose=False):
        start,end,chrom,build,variant_class,isoforms,hgncId,gene_start,gene_end,ref_allele,strand,mutation,barcode = self.data[idx]

        # ref_gene = ref_genome[gene_start-1:gene_end]
        ref_gene = self.seqs[(build,chrom)][hgncId]

        reverse = strand=='-'

        # mutation = self.getMutation(tumorSeqAllele1,tumorSeqAllele2,ref_allele,variant_class)

        start_zero = start - gene_start + 1
        end_zero = end - gene_start + 1
        mutated_gene = self.mutate(mutation,start_zero,end_zero,1,gene_end - gene_start + 1,variant_class,ref_gene,ref_allele)

        #isoforms mutated
        isoform_mutated_seqs = {}
        isoform_seqs = {}
        all_exons_seq = ''
        all_exons_mutated_seq = ''
        for idx,isoform in isoforms.items():
            isoform_mutated_seqs[idx] = ''
            isoform_seqs[idx] = ''
            isoform_is_mutated = False
            for exon_start,exon_end in isoform:
                exon_start_zero = exon_start - gene_start + 1
                exon_end_zero = exon_end - gene_start + 1
                isoform_seqs[idx]+=ref_gene[exon_start_zero-1:exon_end_zero]
                all_exons_seq += ref_gene[exon_start_zero-1:exon_end_zero]
                if exon_start <= start and exon_end >= end:
                    isoform_is_mutated = True
                    start_zero = start - gene_start + 1
                    end_zero = end - gene_start + 1
                    mutated_exon = self.mutate(mutation,start_zero,end_zero,exon_start_zero,exon_end_zero,variant_class,ref_gene,ref_allele)
                    
                    if verbose:
                        print(ref_gene[exon_start_zero-1:exon_end_zero])
                        print(mutated_exon)
                        
                    isoform_mutated_seqs[idx]+=mutated_exon
                    all_exons_mutated_seq += mutated_exon
                else: 
                    isoform_mutated_seqs[idx]+=ref_gene[exon_start_zero-1:exon_end_zero]
                    all_exons_mutated_seq += ref_gene[exon_start_zero-1:exon_end_zero]
            if not isoform_is_mutated: del isoform_mutated_seqs[idx] 
        
        if verbose:
            print(ref_gene[start_zero-20:end_zero+20])
            print(mutated_gene[start_zero-20:end_zero+20])

        if reverse:
            try:
                ref_gene = self.rc(ref_gene)
            except: print('ref_gene:',ref_gene)
            isoform_mutated_seqs = {k:self.rc(v) for k,v in isoform_mutated_seqs.items()}
            isoform_seqs = {k:self.rc(v) for k,v in isoform_seqs.items()}
            all_exons_seq = self.rc(all_exons_seq)
            all_exons_mutated_seq = self.rc(all_exons_mutated_seq)

        isoforms_mutated_aa = {k:str(Seq(v).translate()) for k,v in isoform_mutated_seqs.items()}
        isoforms_aa = {k:str(Seq(v).translate()) for k,v in isoform_seqs.items()}
        all_exons_aa = str(Seq(all_exons_seq).translate())
        all_exons_mutated_aa = str(Seq(all_exons_mutated_seq).translate())

        # filter synonymous mutations
        for k in list(isoforms_mutated_aa.keys()):
            iso_aa = isoforms_mutated_aa[k]
            ref_aa = isoforms_aa[k]
            if iso_aa == ref_aa:
                del isoforms_mutated_aa[k]
        
        if all_exons_aa == all_exons_mutated_aa:
            all_exons_mutated_aa = None

        # return ref_gene, mutated_gene, isoform_mutated_seqs, isoform_seqs, all_exons_seq, all_exons_mutated_seq, isoforms_mutated_aa, isoforms_aa, all_exons_aa, all_exons_mutated_aa
        return isoforms_mutated_aa, isoforms_aa, all_exons_mutated_aa, all_exons_aa, barcode

    #create an ordered set of sequences (using dict as ordered set)
    #map each mutation idx to the ordered set
    def getAllMutations(self):
        exons_concat_ref = orderedSet()
        exons_concat_mut = orderedSet()
        isos_ref = orderedSet(item_is_dict=True)
        isos_mut = orderedSet(item_is_dict=True)
        canonical_ref = orderedSet()
        canonical_mut = orderedSet()

        #0:concat_ref
        #1:concat_mut
        #2:iso_ref
        #3:iso_mut
        #4:canonical_ref
        #5:canonical_mut
        #6:data
        tumors = {}

        getRefAndMutatedSeq = self.getRefAndMutatedSeq

        for idx in tqdm(range(len(self.data))):
            isoforms_mutated_aa, isoforms_aa, all_exons_mutated_aa, all_exons_aa, barcode = getRefAndMutatedSeq(idx)

            isos_ref_idx = isos_ref.add_to_set(isoforms_aa)
            isos_mut_idx = isos_mut.add_to_set(isoforms_mutated_aa)

            canonical_ref_aa = isoforms_aa['1'] if '1' in isoforms_aa else None
            canonical_mut_aa = isoforms_mutated_aa['1'] if '1' in isoforms_mutated_aa else None
            canonical_mut_idx = canonical_mut.add_to_set(canonical_mut_aa)
            canonical_ref_idx = canonical_ref.add_to_set(canonical_ref_aa)

            exons_concat_ref_idx = exons_concat_ref.add_to_set(all_exons_aa)
            exons_concat_mut_idx = exons_concat_mut.add_to_set(all_exons_mutated_aa)

            if tumors.get(barcode)is None: tumors[barcode] = []
            tumors[barcode].append([exons_concat_ref_idx, exons_concat_mut_idx, isos_ref_idx, isos_mut_idx, canonical_ref_idx, canonical_mut_idx, idx])

        ordered_lists = [i.return_ordered_list() for i in [isos_ref,isos_mut,exons_concat_ref,exons_concat_mut, canonical_ref, canonical_mut]]
        return ordered_lists, tumors
    
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




