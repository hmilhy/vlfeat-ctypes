from __future__ import print_function

from collections import namedtuple
from ctypes import (c_int, c_float, c_double, c_void_p,
                    c_ubyte,
                    POINTER, CFUNCTYPE, cast,Array)

import numpy as np
import numpy.ctypeslib as npc
import math

import pdb

from vlfeat.vl_ctypes import (LIB, CustomStructure, Enum,
                        vl_type, vl_size, vl_uint8,
                        np_to_c_types, c_to_vl_types)

from .ikmeans import (VLIKMFilt, VLIKMFilt_p,vl_ikmacc_t,
                      vl_ikm_get_K, vl_ikm_get_ndims, vl_ikm_get_centers,
                      IKMAlgorithm)




class VLHIKMNode(CustomStructure):
    pass

VLHIKMNode_p = POINTER(VLHIKMNode)

VLHIKMNode._fields_ = [
    ('filter',   VLIKMFilt_p),
    ('children', POINTER(VLHIKMNode_p) ),
]


class VLHIKMTree(CustomStructure):
    _fields_ = [
        ('M',          vl_size),
        ('K',          vl_size),
        ('depth',      vl_size),
        ('max_niters', vl_size),
        ('method',     c_int),
        ('verb',       c_int),
        ('root',       VLHIKMNode_p),
    ]

VLHIKMTree_p = POINTER(VLHIKMTree)

##################################################################
#
# create and destroy
vl_hikm_new = LIB['vl_hikm_new']
vl_hikm_new.restype  = VLHIKMTree_p
vl_hikm_new.argtypes = [IKMAlgorithm]

vl_hikm_delete = LIB['vl_hikm_delete']
vl_hikm_delete.restype = None
vl_hikm_delete.argtypes = [VLHIKMTree_p]

# retrieve data and parameters
vl_hikm_get_ndims = LIB['vl_hikm_get_ndims']
vl_hikm_get_ndims.restype  = vl_size
vl_hikm_get_ndims.argtypes = [VLHIKMTree_p]

vl_hikm_get_K = LIB['vl_hikm_get_K']
vl_hikm_get_K.restype  = vl_size
vl_hikm_get_K.argtypes = [VLHIKMTree_p]

vl_hikm_get_depth = LIB['vl_hikm_get_depth']
vl_hikm_get_depth.restype  = vl_size
vl_hikm_get_depth.argtypes = [VLHIKMTree_p]

vl_hikm_get_verbosity = LIB['vl_hikm_get_verbosity']
vl_hikm_get_verbosity.restype  = c_int
vl_hikm_get_verbosity.argtypes = [VLHIKMTree_p]

vl_hikm_get_max_niters = LIB['vl_hikm_get_max_niters']
vl_hikm_get_max_niters.restype  = vl_size
vl_hikm_get_max_niters.argtypes = [VLHIKMTree_p]

vl_hikm_get_root = LIB['vl_hikm_get_root']
vl_hikm_get_root.restype  = VLHIKMNode
vl_hikm_get_root.argtypes = [VLHIKMTree_p]


# set parameters
vl_hikm_set_verbosity = LIB['vl_hikm_set_verbosity']
vl_hikm_set_verbosity.restype  = None
vl_hikm_set_verbosity.argtypes = [VLHIKMTree_p, c_int]

vl_hikm_set_max_niters = LIB['vl_hikm_set_max_niters']
vl_hikm_set_max_niters.restype  = None
vl_hikm_set_max_niters.argtypes = [VLHIKMTree_p, c_int]


# Process data
vl_hikm_init = LIB['vl_hikm_init']
vl_hikm_init.restype  = None
vl_hikm_init.argtypes = [VLHIKMTree_p, vl_size, vl_size, vl_size]

vl_hikm_train = LIB['vl_hikm_train']
vl_hikm_train.restype  = None
vl_hikm_train.argtypes = [VLHIKMTree_p, c_void_p, vl_size]

vl_hikm_push = LIB['vl_hikm_push']
vl_hikm_push.restype  = None
vl_hikm_push.argtypes = [VLHIKMTree_p, c_void_p, c_void_p, vl_size]





##################################################################
def vl_hikmeans(data, K, nleaves,
                max_iters=200, method='lloyd', verbosity=0):
    '''
    tree, asgn = vl_hikmenas(data, K, nleaves) 
    
    tree: is a structure representation the hierachical cluster
    eacho node of the tree os also a dict with fields
    
      depth :: (only at the root node)

      center:: K cluster centers

      sub   :: array of K node structures representing subtrees

    asgn: is a matrix with one column per datum 
    
    '''    
    data = np.asarray(data)
    c_dtype=np_to_c_types.get(data.dtype, None)
    data_p = data.ctypes.data_as(c_void_p)
    
    if c_dtype != c_ubyte:
        raise TypeError('data should be uint8')
    vl_dtype = c_to_vl_types[c_dtype]
    
    M = data.shape[0] # number of components
    N = data.shape[1] # number of elements
    
    algorithm = IKMAlgorithm._members[method.upper()]


    ###################################
    # DO the job
    
    depth = math.ceil(math.log(nleaves)/ \
                      math.log(K))
    depth = int(max(1,depth))

    hikmeans_p = vl_hikm_new(algorithm)
    hikmeans = hikmeans_p.contents
    
    if verbosity:
        print('hikmeans: # dims: %d'%(M))
        print('hikmeans: # data: %d'%(N))
        print('hikmeans: K   : %d'%(K))
        print('hikmeans: depth: %d'%(depth))

    try:
        hikmeans.verb = verbosity
        vl_hikm_init(hikmeans_p, M, K, depth)
        vl_hikm_train(hikmeans_p, data_p, N)
        
        tree = hikm_to_python(hikmeans)

        asgn = np.zeros([hikmeans.depth, N],
                        dtype=np.uint32)
        asgn_p = asgn.ctypes.data_as(c_void_p)
        vl_hikm_push(hikmeans_p, asgn_p, data_p, N)

        if verbosity:
            print('hikmeans: done')
        
        return tree, asgn, hikmeans
    finally:
        #vl_hikm_delete(hikmeans_p)
        pass
        
######################################
# help function
def hikm_to_python(tree):
    ptree = {'K'       : tree.K,
             'depth'   : tree.depth,
             'centers' : [],
             'sub'     : []
    }
    if tree.root != None:
        xcreate(ptree, tree.root.contents)

    return ptree

def xcreate(pnode, node):
    node_filt = cast(node.filter, VLIKMFilt_p).contents
    
    node_K = node_filt.K
    M = node_filt.M
    centers_p = cast(node_filt.centers, POINTER(vl_ikmacc_t))
    centers = np.ctypeslib.as_array(centers_p, (M, node_K)).copy()

    pnode['centers'] = centers

    if node.children:
        arr_sub = []
        [arr_sub.append({'centers':[] , 'sub': []}) for n in range(node_K)]
        
        for k in range(node_K):
            #node.children[0]
            child = cast(node.children[k], VLHIKMNode_p).contents
            xcreate(arr_sub[k], child)
            
        pnode['sub'] = arr_sub
    return pnode




                      
