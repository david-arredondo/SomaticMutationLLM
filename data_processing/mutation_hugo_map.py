
import numpy as np
from saveAndLoad import *
import pandas as pd

mutations = pd.read_csv('./Genie15/data_mutations_extended.txt', sep='\t')

hg19 = pd.read_csv('hg19.bed',sep='\t',header=None)
hg19.columns = ['chrom','chromStart','cromEnd','name','score','strand','thickStart','thickEnd','itemRgb']
hg19.head()

hg38 = pd.read_csv('hg38.bed',sep='\t',header=None)
hg38.columns = ['chrom','chromStart','cromEnd','name','score','strand','thickStart','thickEnd','itemRgb']

beds = {'GRCh37':hg19.values,'GRCh38':hg38.values}

mcolidx = {i:ni for ni,i in enumerate(mutations.columns)}
bcolidx = {i:ni for ni,i in enumerate(hg19.columns)}
mutation_location_gene_map = {}
mutation_location_idx_map = {}

mutations_values = mutations.values

not_found = 0
for m_idx,m in enumerate(mutations_values):
    m_start = m[mcolidx['Start_Position']]
    m_end = m[mcolidx['End_Position']]
    m_chr = str(m[mcolidx['Chromosome']])
    sym = m[mcolidx['Hugo_Symbol']]
    build = m[mcolidx['NCBI_Build']]
    key = (m_chr,m_start,m_end,build)
    if mutation_location_gene_map.get(key) is None: mutation_location_gene_map[key] = []
    if mutation_location_idx_map.get(key) is None: mutation_location_idx_map[key] = []
    mutation_location_idx_map[key].append(m_idx)
    # print(m_start,m_end,m_chr)
    found=0
    for gene in beds[build]:
        b_start = gene[bcolidx['chromStart']]
        b_end = gene[bcolidx['cromEnd']]
        b_chr = gene[bcolidx['chrom']].strip('chr')
        b_name = gene[bcolidx['name']]
        val = (b_name,b_start,b_end)
        # print(b_start,b_end,b_chr)
        if b_chr == m_chr:
            if m_start >= b_start and m_end <= b_end:
                if val not in mutation_location_gene_map[key]:
                    mutation_location_gene_map[key].append(val)
                found=1
    if m_idx%10000==0: print(m_idx,'of',len(mutations.values))

pickleSave(mutation_location_gene_map,'./','mutation_location_gene_map.pkl')
pickleSave(mutation_location_idx_map,'./','mutation_location_idx_map.pkl')
