import numpy as np
from saveAndLoad import *
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

mutations = pd.read_csv('./Genie15/data_mutations_extended.txt', sep='\t')
genomic = pd.read_csv('./Genie15/genomic_information.txt', sep='\t')
clinical_patient = pd.read_csv('./Genie15/data_clinical_patient.txt', sep='\t',skiprows=4)
clinical_sample = pd.read_csv('./Genie15/data_clinical_sample.txt', sep='\t', skiprows=4)

patients = {}
pcolidx = {i:ni for ni,i in enumerate(clinical_patient.columns)}
scolidx = {i:ni for ni,i in enumerate(clinical_sample.columns)}
for idx_p,p in enumerate(tqdm(clinical_patient.values)):
    pid = p[pcolidx['PATIENT_ID']]
    if patients.get(pid) is None: patients[pid] = []
    for idx_s, s in enumerate(clinical_sample.values):
        pid_sample = s[scolidx['PATIENT_ID']]
        sid = s[scolidx['SAMPLE_ID']]
        if pid == pid_sample:
            patients[pid].append(idx_s)

pickleSave(patients,'./','patient_to_tumor_mapping.pkl')