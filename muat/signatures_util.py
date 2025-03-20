import pandas as pd
from tqdm import tqdm
import numpy as np  

def enumerate_150_mutation_types():
    """
    Return a list of all 150 mutation types:
      - 6 single-base changes
      - 48 double-base contexts
      - 96 triple-base contexts
    in strings like:
      'T>C'       # single
      'AT>C'      # double, 5' base
      'T>CG'      # double, 3' base
      'AT>CG'     # triple (5' + mutation + 3')
    """

    # We assume the 'pyrimidine-based' set of ref->alt:
    #   C->A, C->G, C->T, T->A, T->C, T->G
    single_changes = [
        ("C", "A"),
        ("C", "G"),
        ("C", "T"),
        ("T", "A"),
        ("T", "C"),
        ("T", "G"),
    ]
    # Possible flanking bases
    neighbors = ["A", "C", "G", "T"]

    # 1) Enumerate 6 single-base changes.
    singles = []
    for (ref, alt) in single_changes:
        singles.append(f"{ref}>{alt}")

    # 2) Enumerate 48 doubles:
    #    We split them into 24 "5' base + mutation" and
    #    24 "mutation + 3' base".
    doubles = []

    # 2a) 5′ neighbor + mutation, e.g. "AT>C"
    for five_prime in neighbors:
        for (ref, alt) in single_changes:
            doubles.append(f"{five_prime}{ref}>{alt}")

    # 2b) mutation + 3′ neighbor, e.g. "T>CG"
    for (ref, alt) in single_changes:
        for three_prime in neighbors:
            doubles.append(f"{ref}>{alt}{three_prime}")

    # 3) Enumerate 96 triples: "5' base + mutation + 3' base"
    #    e.g. "AT>CG" means five_prime="A", (ref="T", alt="C"), three_prime="G".
    triples = []
    for five_prime in neighbors:
        for (ref, alt) in single_changes:
            for three_prime in neighbors:
                triples.append(f"{five_prime}{ref}>{alt}{three_prime}")

    # Combine them: 6 + 48 + 96 = 150 total
    all_150 = singles + doubles + triples
    return all_150

def to_snv150(tumors, barcode, data, seqs, ref_aa, mut_aa, return_genes = False):
    snv150 = enumerate_150_mutation_types()
    snv150_indexes = {snv:i for i,snv in enumerate(snv150)}
    instances = []
    counts = np.array([0 for i in range(150)])
    comp = {'A':'T','T':'A','C':'G','G':'C'}
    genes = []

    # Grab the indexes from the tumors structure
    data_idxs = [i[-1] for i in tumors[barcode]]
    ref_idxs = [i[4] for i in tumors[barcode]]
    mut_idxs = [i[5] for i in tumors[barcode]]

    for data_idx, ref_idx, mut_idx in zip(data_idxs, ref_idxs, mut_idxs):
        start,end,chrom,build,variant_class,isoforms,hgncId,gene_start,gene_end,ref_allele,strand,mutation,bc = data[data_idx]

        # Pull reference and mutant amino acids
        ref_seq, mut_seq = ref_aa[ref_idx], mut_aa[mut_idx]
        if (ref_seq is None) or (mut_seq is None):
            continue
        if '*' in ref_seq: 
            continue

        # Prepare the position, ref, alt, etc.
        pos = start
        ref = ref_allele
        alt = mutation
        
        # Retrieve the entire reference DNA sequence for this gene
        gene_dna_seq = seqs[(build, chrom)][hgncId]

        # Sanity-check that the base in gene_dna_seq matches ref_allele
        if (len(ref) == 1) and (len(alt) == 1):
            r = pos - gene_start
            five,ref,three = gene_dna_seq[r-1:r+2]
            assert ref == ref_allele, (five,ref,three)
        else: continue

        if ref in 'AG':
            five,ref,alt,three = tuple(map(comp.get, [five,ref,alt,three]))

        triple = f'{five}{ref}>{alt}{three}'
        single = triple[1:4]
        double_5 = triple[:4]
        double_3 = triple[1:]

        instances+=[single, double_5, double_3, triple]
        genes.append(hgncId)
    
    for i in instances:
        counts[snv150_indexes[i]]+=1

    if return_genes:
        return counts, genes
    
    return counts

def get_snv150_df(class_data, data, tumors, seqs, ref_aa, mut_aa, return_somatt_df = False):
    barcodes = class_data['barcode'].values
    snv150_data = {}
    for barcode in tqdm(barcodes):
        mut_type_vector = to_snv150(tumors, barcode, data, seqs, ref_aa, mut_aa)
        snv150_data[barcode] = mut_type_vector
    snv150_df = pd.DataFrame(snv150_data).T
    snv150_df.columns = enumerate_150_mutation_types()
    if return_somatt_df:
        somatt_df = class_data[class_data['barcode'].isin(snv150_df.index)]
        return snv150_df, somatt_df
    return snv150_df

def get_gene_indexes(class_data, tumors, data, seqs, ref_aa, mut_aa):
    barcodes = class_data['barcode'].values
    all_genes=set()
    for barcode in barcodes:
        _, genes = to_snv150(tumors, barcode, data, seqs, ref_aa, mut_aa, return_genes=True)
        all_genes.update(genes)
    gene_indexes = {gene:i for i,gene in enumerate(all_genes)}
    return gene_indexes

def get_snv150_withPos_df(class_data, data, tumors, seqs, ref_aa, mut_aa, return_somatt_df = False):
    barcodes = class_data['barcode'].values
    snv150_data = {}

    gene_indexes = get_gene_indexes(class_data, tumors, data, seqs, ref_aa, mut_aa)

    for barcode in tqdm(barcodes):
        gene_counts = [0 for i in range(len(gene_indexes))]
        mut_type_vector, genes = to_snv150(tumors, barcode, data, seqs, ref_aa, mut_aa, return_genes=True)
        for g in genes: gene_counts[gene_indexes[g]]+=1
        total_vector = np.concatenate([mut_type_vector, gene_counts])
        snv150_data[barcode] = total_vector

    snv150_df = pd.DataFrame(snv150_data).T
    snv150_df.columns = enumerate_150_mutation_types() + list(gene_indexes.keys())
    if return_somatt_df:
        somatt_df = class_data[class_data['barcode'].isin(snv150_df.index)]
        return snv150_df, somatt_df
    return snv150_df