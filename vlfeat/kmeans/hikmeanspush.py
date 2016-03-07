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

from vlfeat.ikmeans import (VLIKMFilt, VLIKMFilt_p,
                      vl_ikm_get_K, vl_ikm_get_ndims, vl_ikm_get_centers,
                      vl_ikm_new, IKMAlgorithm,vl_ikm_init)

from vlfeat.hikmeans import (VLHIKMNode, VLHIKMTree, VLHIKMNode_p,
                       VLHIKMNode_p, IKMAlgorithm,
                       vl_hikm_get_depth,vl_hikm_get_K,
                       vl_hikm_get_ndims,vl_hikm_delete,
                       vl_hikm_push)


def vl_hikmeanspush(tree, data, method='lloyd', verbosity=0):
    data = np.asarray(data)
    M,N = data.shape
    
    algorithm = IKMAlgorithm._members[method.upper()]

    hikmeans=python_to_hikm(tree, algorithm)

    depth=vl_hikm_get_depth(hikmeans)

    if verbosity:
        print("vl_hikmeanspush: ndim, %d K:%d depth: %d"%
              (vl_hikm_get_ndims(tree),
               vl_hikm_get_K(tree),
               depth))

    ids = np.zeros([depth, N], dtype=np.uint32)
    ids_p = np.ctypeslib.as_ctypes(ids)

    data_p = np.ctypeslib.as_ctypes(data)

    pdb.set_trace()
    vl_hikm_push(hikmeans, ids_p, data_p, N)
    #vl_hikm_delete(tree)
    for id_ in ids:
        id_ = id_+1
    return ids, hikmeans

def python_to_hikm(ptree, algorithm):
    pK = ptree['K']
    pdepth = ptree['depth']

    
    tree       = VLHIKMTree
    tree.depth = pdepth
    tree.K     = pK
    tree.M     = 0
    tree.method= algorithm
    tree.root  = xcreate(tree, ptree)
    pdb.set_trace()
    return tree

def xcreate(tree, pnode):
    #pdb.set_trace()
    pcenters = pnode['centers']
    pcenters_p = np.ctypeslib.as_ctypes(pcenters)
    psub = pnode['sub']

    M , node_K = pcenters.shape
    if M==0:
        print('A NODE.CENTERS has zero rows')

    if node_K > tree.K:
        print('A NODE.centers has more coluns than overall clusters TREE.K')    

    if (tree.M == 0):
        tree.M = M
    elif(M != tree.M):
        print("A NODE.CENTERS fields has inconsistent dimensionality")
    
    node = VLHIKMNode
    node.filter=vl_ikm_new(tree.method)
    node.children=None

    vl_ikm_init(node.filter, pcenters_p, M, node_K)

    #pdb.set_trace()

    
    if len(psub) != 0:
        node_children = (VLHIKMNode*node_K)()
        node_children_p = cast(node_children, POINTER(VLHIKMNode))
        node.children  = node_children_p
        pdb.set_trace()
        for k in range(node_K):
            node_children_p[k] = xcreate(tree, psub[k])
    return node
