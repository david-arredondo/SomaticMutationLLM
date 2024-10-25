# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.neighbors import NearestNeighbors
# from scipy.spatial.distance import pdist, squareform
# from sklearn.cluster import DBSCAN
# from sklearn.cluster import KMeans
# from saveAndLoad import *

# output = pickleLoad('tsne_output.pkl')
# metric,embeddings,predictions,time_event,labels,detailed, X_tsne  = output[('somatt','survival')]
# snn_labs = {}

# # for k in [14,15,16,17,18]:
# for k in [14]:
#     print('fitting')
#     nn = NearestNeighbors(n_neighbors=k)
#     nn.fit(X_tsne)
#     print('nn.kneighbors')
#     distances, indices = nn.kneighbors(X_tsne)

#     snn_graph = np.zeros((len(X_tsne), len(X_tsne)))
#     print('loop')
#     print(len(X_tsne))
#     for i in range(len(X_tsne)):
#         for j in range(i+1, len(X_tsne)):
#             shared_neighbors = len(np.intersect1d(indices[i], indices[j]))
#             snn_graph[i, j] = shared_neighbors
#             snn_graph[j, i] = shared_neighbors

#     min_shared_neighbors = 3
#     eps = 0.5
#     snn_distances = 1/(snn_graph + 1e-6)
#     print('clustering')
#     clustering = DBSCAN(eps=eps, min_samples=min_shared_neighbors, metric='precomputed')
#     labels = clustering.fit_predict(snn_distances)
#     snn_labs[k] = labels
#     pickleSave(snn_labs, './','snn_labs.pkl')
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor
from saveAndLoad import pickleLoad, pickleSave

def compute_shared_neighbors(i, indices, X_len):
    shared_neighbors = np.zeros(X_len)
    for j in range(i+1, X_len):
        shared_neighbors[j] = len(np.intersect1d(indices[i], indices[j]))
    return i, shared_neighbors

# Create a standalone function for multiprocessing
def process_neighbors(args):
    return compute_shared_neighbors(*args)

def parallel_snn_computation(X_tsne, indices):
    X_len = len(X_tsne)
    snn_graph = np.zeros((X_len, X_len))

    num_workers = os.cpu_count() or 1  # Get number of CPU cores, fallback to 1 if None
    
    # Prepare arguments for the parallel map
    args = [(i, indices, X_len) for i in range(X_len)]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_neighbors, args))
    
    # Fill the snn_graph matrix with the results
    for i, shared_neighbors in results:
        snn_graph[i, i+1:] = shared_neighbors[i+1:]
        snn_graph[i+1:, i] = shared_neighbors[i+1:]
    
    return snn_graph

output = pickleLoad('tsne_output.pkl')
metric, embeddings, predictions, time_event, labels, detailed, X_tsne = output[('somatt', 'survival')]
snn_labs = {}

for k in [9]:
    print(k)
    for msn in [10,11,12,13,14,15]:
        print(k,msn)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_tsne)
        distances, indices = nn.kneighbors(X_tsne)

        # Use parallel process-based computation
        snn_graph = parallel_snn_computation(X_tsne, indices)

        min_shared_neighbors = msn
        eps = 16
        snn_distances = 1 / (snn_graph + 1e-6)
        clustering = DBSCAN(eps=eps, min_samples=min_shared_neighbors, metric='precomputed')
        labels = clustering.fit_predict(snn_distances)
        snn_labs[(k,msn)] = labels
        pickleSave(snn_labs, './', 'snn_labs.pkl')


