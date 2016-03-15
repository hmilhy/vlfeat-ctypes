from __future__ import print_function

from collections import namedtuple
from ctypes import (c_int, c_float, c_double, c_void_p,
                    c_ubyte,c_uint32,
                    POINTER, CFUNCTYPE, cast)
import numpy as np
import numpy.ctypeslib as npc
import math

import pdb

from vlfeat.vl_ctypes import (LIB, CustomStructure, Enum,
                        vl_type, vl_size, vl_uint8,
                        vl_int32,vl_uint32,vl_uint,
                        np_to_c_types, c_to_vl_types)

vl_ikmacc_t = vl_int32

class IKMAlgorithm(Enum):
    LLOYD = 0
    ELKAN = 1

class VLIKMFilt(CustomStructure):
    _fields_ =[
        ('M',          vl_size),
        ('K',          vl_size),
        ('max_niters', vl_size),
        ('method',     c_int),
        ('verb',       c_int),
        ('centers',    POINTER(vl_ikmacc_t)),
        ('inter_dist', POINTER(vl_ikmacc_t)),
    ]

VLIKMFilt_p = POINTER(VLIKMFilt)

# craete and destroy
vl_ikm_new = LIB['vl_ikm_new']
vl_ikm_new.restype=VLIKMFilt_p
vl_ikm_new.argtypes=[IKMAlgorithm]

vl_ikm_delete = LIB['vl_ikm_delete']
vl_ikm_delete.restype = None
vl_ikm_delete.argtypes = [VLIKMFilt_p]

# process data
vl_ikm_init = LIB['vl_ikm_init']
vl_ikm_init.restype  = None
vl_ikm_init.argtypes = [VLIKMFilt_p,c_void_p,vl_size, vl_size]

vl_ikm_init_rand = LIB['vl_ikm_init_rand']
vl_ikm_init_rand.restype = None
vl_ikm_init_rand.argtypes = [VLIKMFilt_p, vl_size, vl_size]

vl_ikm_init_rand_data = LIB['vl_ikm_init_rand_data']
vl_ikm_init_rand_data.restype = None
vl_ikm_init_rand_data.argtypes = [VLIKMFilt_p, POINTER(vl_uint8), vl_size, vl_size, vl_size]

vl_ikm_train = LIB['vl_ikm_train']
vl_ikm_train.restype = None
vl_ikm_train.argtypes = [VLIKMFilt_p, POINTER(vl_uint8), vl_size]

vl_ikm_push = LIB['vl_ikm_push']
vl_ikm_push.restype = c_int
vl_ikm_push.argtypes = [VLIKMFilt_p, POINTER(vl_uint32), POINTER(vl_uint8), vl_size]


def vl_ikmeans(X, K, max_niters=200 ,method='LloyD', verbosity=0):
    M,N=X.shape
    ######################################################
    # Do the job
    if verbosity==1:
        print('vl_ikmeans: maxInters%d\n'%(max_niters))

    #pdb.set_trace()
    data = np.asarray(X, dtype=np.uint8)
    data_p = data.ctypes.data_as(c_void_p)
    data_p = cast(data_p, POINTER(c_ubyte))
    algorithm = IKMAlgorithm._members[method.upper()]
    
    ikmf = vl_ikm_new(algorithm)
    ikm = ikmf.contents
    
    ikm.verbosity = verbosity
    ikm.max_niters = max_niters
    vl_ikm_init_rand_data(ikmf, data_p, M,N,K)
    
    err = vl_ikm_train(ikmf,data_p, N)
    
    #pdb.set_trace()
    #####################################################
    # Return results
    center_p = ikm.centers
    C = np.ctypeslib.as_array(center_p, (K,M)).copy()
    
    I = np.zeros(N, dtype=np.uint32)
    I_p = I.ctypes.data_as(c_void_p)
    I_p = cast(I_p, POINTER(c_uint32))
    
    vl_ikm_push(ikmf, I_p, data_p, N)

    return C, I
    
