
import numpy as np
from vlfeat.misc.vl_binsum import vl_binsum

import pdb

def vl_hikmeanshist(tree, path):
    '''
    computes the histogram of the HIKM tree
    nodes activited by the root-to-leaf paths
    PATH is usually obtained by quantizing data
    by means of vl_HIKMEANSPUSH()

    The histogram H has bin for each node of the HIKM tree.
    The tree has K = TREE.K nodes and depth D=tree.depth.
    Therefore there are M=(K^(D+1)-1)/(K-1) nodes in the tree
    (not counting the root which carries no information)
    Nodes are stacked into a vector of bins in breadth first order

    Example:
      H[0] = # of paths such that PATH[0,:] = 1
      H[K] = # of paths such that PATH[0,:] = K

    
    '''
    #pdb.set_trace()
    
    K = tree['K']
    D = tree['depth']
    M = int((K**(D+1)-1)/(K-1))

    hist = np.zeros([M, 1], dtype=np.int)
    p = np.zeros(path.shape[1], dtype=np.int)

    hist[0] = path.shape[1]

    #pdb.set_trace()

    
    for d in range(D):
        p = p*K + path[d,:]
        #pdb.set_trace()
        
        hist = vl_binsum(hist, 1, p)                                               

    return hist
