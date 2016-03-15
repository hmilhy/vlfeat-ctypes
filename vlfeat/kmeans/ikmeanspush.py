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

from .ikmeans import (IKMAlgorithm,VLIKMFilt,vl_ikmacc_t,
                      vl_ikm_new, 
                      vl_ikm_init, vl_ikm_push)

def vl_ikmeanspush(X, C, verbosity=0, method='LloyD'):
    #pdb.set_trace()
    M,N = X.shape
    K = C.shape[1]

    C_p = cast(C.ctypes.data_as(c_void_p), POINTER(vl_ikmacc_t))
    data_p = cast(X.ctypes.data_as(c_void_p), POINTER(vl_uint8))
    
    asgn = np.zeros(N, dtype=np.uint32)
    asgn_p = cast(asgn.ctypes.data_as(c_void_p), POINTER(c_uint32) )

    algorithm = IKMAlgorithm._members[method.upper()]
    
    ikmf = vl_ikm_new(algorithm)
    
    ikmf.verbosity = verbosity
    vl_ikm_init(ikmf, C_p,M, K)
    vl_ikm_push(ikmf, asgn_p, data_p, N)

    
    return asgn
