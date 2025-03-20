from SigProfilerMatrixGenerator import install as genInstall
from SigProfilerAssignment import Analyzer as Analyze
import os
from saveAndLoad import *
import pandas as pd

matrixes= pickleLoad('/data/dandreas/SomaticMutationsLLM/matrix_10_percent.pkl')
output = '/data/dandreas/SomaticMutationsLLM/cosmic_fit/'
failed = []
for i in matrixes['96'].columns:
    try:
        sample = pd.DataFrame(matrixes['96'][i])
        sample_name = i.split('.')[0]
        saveFolder = output+sample_name+'/'
        os.makedirs(saveFolder, exist_ok=False)
        Analyze.cosmic_fit(sample, saveFolder, input_type="matrix", context_type="96",
                    collapse_to_SBS96=False, cosmic_version=3.4, exome=False,
                    genome_build="GRCh37", signature_database=None,
                    exclude_signature_subgroups=None, export_probabilities=False,
                    export_probabilities_per_mutation=False, make_plots=False,
                    sample_reconstruction_plots=False, verbose=False)
    except:
        failed.append(i)
print(len(failed))
failed


