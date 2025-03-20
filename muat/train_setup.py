import pandas as pd
from io import StringIO
import gzip
import numpy as np
from saveAndLoad import *
import sys
from torch.utils.data import Dataset
import torch.nn as nn
if '../model/' not in sys.path: sys.path.append('../model/')
from models import negative_log_partial_likelihood, getPatientGroupedLoaders, c_index, SoMatt_LateFusion
from custom_dataset import *
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch import optim
from dnn import *
from fusion import *
from signatures_util import *

class_data = pd.read_csv('../labeled_data/data_1_00percentMinCancerType.csv')
class_data_msk468 = class_data[class_data['assay']=='MSK-IMPACT468']
survival_data = pd.read_csv('../labeled_data/data_0_00_survival_ratio1.csv')
survival_data_msk468 = survival_data[survival_data['assay']=='MSK-IMPACT468']

tumors = pickleLoad('../aa/tumors.pkl')
data = pickleLoad('../data_processing/consolidated_data.pkl')
seqs = pickleLoad('../data_processing/dna_seq_by_hgncId.pkl')
ref_aa = pickleLoad('../aa/canonical_ref.pkl')
mut_aa = pickleLoad('../aa/canonical_mut.pkl')
mut_embeddings = np.load('../aa/canonical_mut_average_embeddings_esm.npy')
ref_embeddings = np.load('../aa/canonical_ref_embeddings_esm2.npy')
assays = pickleLoad('../aa/assays.pkl')