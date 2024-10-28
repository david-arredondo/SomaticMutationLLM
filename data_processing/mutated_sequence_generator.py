import numpy as np
from saveAndLoad import *
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
import concurrent.futures

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
    
    def mutate(self,mutation,mut_start_zero,mut_end_zero,exon_start_zero,exon_end_zero,variant_class,ref_gene,ref_allele, reverse):

        if variant_class == 'Silent': return ref_gene[exon_start_zero-1:exon_end_zero]

        if variant_class in ['Nonsense_Mutation','Frame_Shift_Del','Frame_Shift_Ins']:
            # ENFORCE STOP CODON
            mutation = 'TAAATAAATAA'
            if reverse: mutation = self.rc(mutation)
        
        l = len(mutation) - len(ref_allele)
        if variant_class == 'Missense_Mutation': assert l == 0
        mutated_gene = ref_gene[:mut_start_zero-1] + mutation + ref_gene[mut_end_zero:]
        mutated_exon = mutated_gene[exon_start_zero-1:exon_end_zero + l]

        return mutated_exon

    def getRefAndMutatedSeq(self,idx,verbose=False):
        start,end,chrom,build,variant_class,isoforms,hgncId,gene_start,gene_end,ref_allele,strand,mutation,barcode = self.data[idx]

        # ref_gene = ref_genome[gene_start-1:gene_end]
        ref_gene = self.seqs[(build,chrom)][hgncId]

        reverse = strand=='-'

        # mutation = self.getMutation(tumorSeqAllele1,tumorSeqAllele2,ref_allele,variant_class)

        start_zero = start - gene_start + 1
        end_zero = end - gene_start + 1
        mutated_gene = self.mutate(mutation,start_zero,end_zero,1,gene_end - gene_start + 1,variant_class,ref_gene,ref_allele,reverse)

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
                    mutated_exon = self.mutate(mutation,start_zero,end_zero,exon_start_zero,exon_end_zero,variant_class,ref_gene,ref_allele,reverse)
                    
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