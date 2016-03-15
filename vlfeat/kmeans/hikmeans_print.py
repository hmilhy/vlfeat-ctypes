from .hikmeans import (VLHIKMNode, VLHIKMTree,
                       VLHIKMNode_p, VLHIKMTree_p)
from .ikmeans import VLIKMFilt, VLIKMFilt_p, vl_ikmacc_t
from ctypes import *
import numpy as np

def hikmeans_print(hikmeans):
    print('===HIKMeans:===')
    print('M: %d'%(hikmeans.M))
    print('K: %d'%(hikmeans.K))
    print('depth:%d'%(hikmeans.depth))
    hikmeans_print_node(hikmeans.root.contents, 0)

def hikmeans_print_node(node, i):
    print('%sHIKMeans node:'%('*'*i))
    node_filt = cast(node.filter, VLIKMFilt_p).contents
    M = node_filt.M
    node_K = node_filt.K
    centers_p =cast(node_filt.centers, POINTER(vl_ikmacc_t))
    centers = np.ctypeslib.as_array(centers_p, (node_K, M))
    print('%scenters:%s'%('*'*i,str(centers)))
    if node.children:
        for k in range(node_K):
            child = cast(node.children[k], VLHIKMNode_p).contents
            hikmeans_print_node(child,  i+1)
    

def hik_py_print(tree):
    print('tree:')
    print('K:%d'%(tree['K']))
    print('depth:%d'%(tree['depth']))
    hik_py_node(tree, 0)
    
def hik_py_node(node, i):
    print('%scenters:%s'%('*'*i,  str(node['centers'])))
    for n in node['sub']:
        hik_py_node(n, i+1)
