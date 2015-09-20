# coding: utf-8
__author__='Michele Buccoli'
__email__ = "michele.buccoli@polimi.it"

"""
This script identifies the structure of a given track using the algorithm
proposed by Kristoffer Jensen in:

    Jensen, K., 
    Multiple scale music segmentation using rhythm, timbre, and harmony
    EURASIP Journal on Advances in Signal Processing, 2007         

The script partly is written after the Music Structural Analysis Framework 
by Oriol Nieto https://github.com/urinieto/msaf/
"""


from scipy.spatial import distance
from scipy.linalg import toeplitz
from scipy.sparse.csgraph import shortest_path
import numpy as np
import logging
import time
import matplotlib.pyplot as plt


def compute_ssm(X, metric="euclidean", normalization=np.max):
    """It computes the self-similarity matrix of X.
    From Nieto implementation of Music Structure Analysis Framework
    https://github.com/urinieto/msaf/
    """
    D = distance.pdist(X, metric=metric)
    D = distance.squareform(D)
    if normalization is not None:
        D/=normalization(D)
    return 1 - D

T = None

def compute_costs(SSM, alpha=0.4, maxcost=np.infty):
    """ It computes the cost from the frame i to the frame j as the sum
    of the sub-matrix SSM[i:j,i:j]. The sum is normalized by the distance 
    (j-i+1) and a segmentation parameter alpha is added.
    """
    costs=np.ones(SSM.shape)*maxcost
    if T is None or T.size<costs.size:
        global T
        T=toeplitz(np.arange(1,costs.shape[0]+1))
        T_=T
    else:
        T_=T[0:costs.shape[0],0:costs.shape[0]]
    SSM/=2
    for v in range(SSM.shape[0]):
        for a in range(v+1,SSM.shape[1]):
            costs[v,a]=SSM[v:a+1,v:a+1].sum()
    costs=(costs/T_)+alpha
    return costs

        
def find_path(predecessors, start, end):
    """ It computes the shortest path from the matrix of predecessors
    """
    new_end=-1;
    path=[end]
    while new_end!=start:
        new_end=predecessors[start,end]
        path.append(new_end)
        end=new_end
    return path[::-1]
    

def compute_boundaries(costs,SSM, F, alpha=0.4):
    dist_matrix, predecessors=shortest_path(costs, return_predecessors=True)    
    path=find_path(predecessors, start=0, end=SSM.shape[0]-1)    
    return path

    
def segment(F, alpha=0.4, SSM=False, metric='euclidean'):
    """Main process for the segmenter proposed by Jensen

    Parameters
    ----------
    F: np.array(N, P)
        Features matrix: N row frames and P features
        if SSM is True:
        F is the Self-Similarity Matrix
    alpha: 
        parameter of penalty for a new segmentation: needs to be tuned in advance
    metric: 
        The metric to be used for the SSM computation

    Returns
    -------
    est_idxs : np.array(K)
        Estimated indeces the segment boundaries in frames.
    est_labels : np.array(K-1)
        Estimated labels for the segments, equals -1 because the algorithm
        only performs the boundary detection
    """       
    if SSM:
        SSM=F
    else:
        SSM=1-compute_ssm(F,metric)

    costs=compute_costs(SSM, alpha)
    est_idxs=compute_boundaries(costs,SSM,F, alpha)
    est_labels = np.ones(len(est_idxs) - 1) * -1

    return est_idxs, est_labels

    