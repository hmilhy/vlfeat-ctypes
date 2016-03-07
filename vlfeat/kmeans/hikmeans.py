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

from vlfeat.ikmeans import (VLIKMFilt, VLIKMFilt_p,
                      vl_ikm_get_K, vl_ikm_get_ndims, vl_ikm_get_centers,
                      IKMAlgorithm)




class VLHIKMNode(CustomStructure):
    _fields_ = [
        ('filter',   VLIKMFilt_p),
        ('children', c_void_p ),
    ]

class VLHIKMTree(CustomStructure):
    _fields_ = [
        ('M',          vl_size),
        ('K',          vl_size),
        ('depth',      vl_size),
        ('max_niters', vl_size),
        ('method',     c_int),
        ('verb',       c_int),
        ('root',       c_void_p),
    ]

VLHIKMTree_p = POINTER(VLHIKMTree)
VLHIKMNode_p = POINTER(VLHIKMNode)
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
                max_iters=200, method='lloyd', verbose=0):
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
    
    if c_dtype != c_ubyte:
        raise TypeError('data should be uint8')
    vl_dtype = c_to_vl_types[c_dtype]
    
    M,N = data.shape

    algorithm = IKMAlgorithm._members[method.upper()]

    depth = int(max(1,math.ceil(math.log(nleaves)/math.log(K))))

    ###################################
    # DO the job
    hikmeans_p = vl_hikm_new(algorithm)
    hikmeans = hikmeans_p[0]
    if verbose:
        print('hikmeans: # dims: %d'%(M))
        print('hikmeans: # data: %d'%(N))
        print('hikmeans: K   : %d'%(K))
        print('hikmeans: depth: %d'%(depth))

    try:
        hikmeans.verb = verbose
        vl_hikm_init(hikmeans_p, M, K, depth)

        data_p = data.ctypes.data_as(c_void_p)
        vl_hikm_train(hikmeans_p, data_p, N)
        
        if verbose:
            print('hikmeans: done')

        tree = hikm_to_python(hikmeans, c_dtype)
        
        return hikmeans,tree
            
    finally:
        #vl_hikm_delete(hikmeans_p)
        pass

        

def hikm_to_python(tree, c_dtype):
    K = vl_hikm_get_K(tree)
    depth = vl_hikm_get_depth(tree)
    
    ptree = {'K'       : K,
             'depth'   : depth,
             'centers' : [],
             'sub'     : []
    }
    if tree.root != None:
        node = cast(tree.root, VLHIKMNode_p).contents
        ptree = xcreate(ptree, node, c_dtype)

    return ptree

def xcreate(pnode, node, c_dtype):
    node_filter_p = cast(node.filter, VLIKMFilt_p)

    node_K = vl_ikm_get_K(node_filter_p)
    M = vl_ikm_get_ndims(node_filter_p)
    
    centers_p = vl_ikm_get_centers(node_filter_p)
    centers_p = cast(centers_p, POINTER(c_dtype))
    centers = np.ctypeslib.as_array(centers_p, (node_K, M)).copy()
    pnode['centers'] = centers
    
    if node.children != None:
        node_children_p = cast(node.children, POINTER(VLHIKMNode_p))
        arr_sub = []
        [arr_sub.append({'centers':[] , 'sub': []}) for n in range(node_K)]
        
        for k in range(node_K):
            node_ = cast(node_children_p[k], VLHIKMNode_p)
            node_ = node_.contents
            
            xcreate(arr_sub[k], node_, c_dtype)
            
            #pdb.set_trace()
            pass
        pnode['sub'] = arr_sub
    #pdb.set_trace()
    return pnode




                      
