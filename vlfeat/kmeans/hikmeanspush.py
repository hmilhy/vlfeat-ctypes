from __future__ import print_function

from collections import namedtuple
from ctypes import (c_int, c_float, c_double, c_void_p,
                    c_ubyte,
                    POINTER, CFUNCTYPE, cast,Array, pointer)

import numpy as np
import numpy.ctypeslib as npc
import math

import pdb

from vlfeat.vl_ctypes import (LIB, CustomStructure, Enum,
                        vl_type, vl_size, vl_uint8,
                        np_to_c_types, c_to_vl_types)

from .ikmeans import (VLIKMFilt, VLIKMFilt_p,
                      vl_ikm_new, IKMAlgorithm,vl_ikm_init)

from .hikmeans import (VLHIKMNode, VLHIKMTree, VLHIKMNode_p,
                       VLHIKMNode_p, IKMAlgorithm,
                       vl_hikm_delete, vl_hikm_push)



def vl_hikmeanspush(tree, data, method='lloyd', verbosity=0):
    #pdb.set_trace()
    data = np.asarray(data, dtype=np.uint8)
    c_dtype = np_to_c_types.get(data.dtype, None)
    if c_dtype !=c_ubyte:
        raise TypeError('DATA must be UINT8')
    
    data_p = data.ctypes.data_as(c_void_p)
    N = data.shape[1]
    
    algorithm = IKMAlgorithm._members[method.upper()]

    #####################################
    # Do the job
    hikmeans = python_to_hikm(tree, algorithm)
    hikmeans.verbosity = verbosity
    hikmeans_p = pointer(hikmeans)

        
    if verbosity:
        print("vl_hikmeanspush: ndim: %d  K:%d  depth: %d"%
              (hikmeans.M,
               hikmeans.K,
               hikmeans.depth))

    asgn = np.zeros([hikmeans.depth, N],
                    dtype=np.uint32)
    asgn_p= asgn.ctypes.data_as(c_void_p)
    

    vl_hikm_push(hikmeans_p, asgn_p, data_p, N)

    #vl_hikm_delete(hikmeans_p)

    #asgn = asgn+1
    if verbosity:
        print('hikmeanspush: done')

        
    return asgn, hikmeans

def python_to_hikm(ptree, algorithm):
    tree = VLHIKMTree(0,
                      ptree['K'],
                      ptree['depth'],
                      0,
                      algorithm.value,
                      0,
                      None)
    tree.root  = pointer(xcreate(tree, ptree))
    return tree

def xcreate(tree, pnode):
    #psub = pnode['sub']
    M = pnode['centers'].shape[1]
    node_K = pnode['centers'].shape[0]

    
    if M==0:
        print('A NODE.CENTERS has zero rows')

    if node_K > tree.K:
        print('A NODE.centers has more coluns than overall clusters TREE.K')    

    if (tree.M == 0):
        tree.M = M
    elif(M != tree.M):
        print("A NODE.CENTERS fields has inconsistent dimensionality")



    node = VLHIKMNode(vl_ikm_new(tree.method), None)
    node.filter=vl_ikm_new(tree.method)
    node.children=None
    
    pcenters = pnode['centers']
    pcenters_p = np.ctypeslib.as_ctypes(pcenters)
    vl_ikm_init(node.filter, pcenters_p, M, node_K)

    psub = pnode['sub']
    if psub:
        node.children = (VLHIKMNode_p*node_K)()
        
        for k in range(node_K):
            node.children[k] = pointer(xcreate(tree, psub[k]))
    
    return node
