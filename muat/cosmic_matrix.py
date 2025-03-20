from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as matGen
from saveAndLoad import *

folder = '/data/dandreas/SomaticMutationsLLM/vcf_for_cosmic_10percent_vcf/'
matrix_output = matGen.SigProfilerMatrixGeneratorFunc("matrix_10_percent",
                                                       "GRCh37",
                                                        folder,
                                                        plot=False, 
                                                        exome=False, 
                                                        bed_file=None, 
                                                        chrom_based=False, 
                                                        tsb_stat=False, 
                                                        seqInfo=False, 
                                                        cushion=100)
pickleSave(matrix_output, '/data/dandreas/SomaticMutationsLLM/', 'matrix_10_percent.pkl')