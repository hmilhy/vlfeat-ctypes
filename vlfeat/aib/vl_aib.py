
from ctypes import (c_void_p, c_double,
                    POINTER, pointer)

from vlfeat.vl_ctypes import (LIB, CustomStructure,
                              vl_type,
                              vl_size, vl_uint)

import numpy as np

class VLAIB(CustomStructure):
    _fields_ = [
        ('nodes',      POINTER(vl_uint)),
        ('nentries',  vl_uint),
        ('beta',        POINTER(c_double)),
        ('bidx',         POINTER(vl_uint)),
        
        ('which',      POINTER(vl_uint)),
        ('nwhich',    vl_uint),

        ('Pcx',          POINTER(c_double)),
        ('Px',            POINTER(c_double)),
        ('Pc',            POINTER(c_double)),
        ('nvalues',   vl_uint),
        ('nlabels',    vl_uint),

        ('parents',   POINTER(vl_uint)),
        ('costs',       POINTER(c_double)),

        ('verbosity', vl_uint),
    ]
VLAIB_p = POINTER(VLAIB)


import pdb


##########################################
vl_aib_new = LIB['vl_aib_new']
vl_aib_new.restype = VLAIB_p
vl_aib_new.argtypes = [POINTER(c_double), vl_uint, vl_uint]

vl_aib_delete=LIB['vl_aib_delete']
vl_aib_delete.restype = None
vl_aib_delete.argtypes = [VLAIB_p]

vl_aib_process = LIB['vl_aib_process']
vl_aib_process.restype = None
vl_aib_process.argtypes = [VLAIB_p]

#vl_aib_get = LIB['vl_aib']
##########################################
def vl_aib(pcx, verbosity =1, cluster_null=False):
    #pdb.set_trace()
    pcx_cpy = pcx.copy()
    pcx_cpy = np.asarray(pcx_cpy, dtype=np.double)
    pcx_p = pcx_cpy.ctypes.data_as(POINTER(c_double))
    
    nlabels = pcx.shape[0]
    nvalues = pcx.shape[1]

    if verbosity:
        print('vl_aib: clustering nul probability variables: %s'%(str(cluster_null)) )
        
    aib_p = vl_aib_new(pcx_p, vl_uint(nvalues), vl_uint(nlabels))
    aib = aib_p.contents
    aib.verbosity = verbosity
    
    vl_aib_process(aib_p)

    parents_p = aib.parents
    costs_p = aib.costs
    
    parents = np.ctypeslib.as_array(parents_p, (2*nvalues-1,)).copy()
    costs = np.ctypeslib.as_array(costs_p, (nvalues,)). copy()

    #vl_aib_delete(aib_p)

    #pdb.set_trace()
    
    if cluster_null:
        parents,costs = cluster_null_nodes(parents, nvalues, costs, verbosity)
    
    # save back parents
    for n in range(2*nvalues-1):
        if parents[n] > 2*nvalues-1:
            parents[n] =0
        else:
            #parents[n] = parents[n]+1
            pass
    return parents, costs

####################################################
def cluster_null_nodes(parents, nvalues, costs, verbosity):
    # cout null nodes so far
    nnull =0
    for n in range(nvalues):
        if parents[n] >= 2*nvalues-1:
            nnull = nnull+1

    if nnull == 0:
        return parents, costs
    a = nvalues
    b = nvalues + nnull -1-1
    c = b+1
    d = c+1
    e = 2*nvalues-2

    dp = nvalues
    ep = 2*nvalus -2 -nnull

    if verbosity:
        print('vl_aib: a:%d, b:%b, c:%c, d:%d, e:%d, dp:%d, ep:%d'%(a,b,c,d,e,dp, ep))

    # search first leaf that has been merged
    first_parent = e
    first = 0
    for n in range(navlues):
        if parents[n] <= e and parents[n] != 1:
            if firtst_parent >= parents[n]:
                first_parent = parents[n]
                first = n

    if verbosity:
        print('vl_aib: nnull:%d, nvalus:%d, first:%d'%(nnull, nvalues, first))

    for n in range(e):
        if parents[n] <= e and parents[n] != 0:
            parents[n] = parents[n] + e - ep

    for n in range(e, d-1, -1):
        parnets[n] = parents[n-(e-ep)]

        
    # find first null node and connect it to a
    last_intermed = a
    n1 = 0
    for n in range(a):
        if parents[n] > e:
            parents[n] = last_intermed
            n1 = n
            break

    if verbosity:
        print('vl_aib: first null %d parents seto to last_intermed:%d'%(n, last_intermed))

    for n in range(n1, n):
       if parents[n] > e:
           parents[n] = last_intermed
           parens[last_intermd] = last_intermed +1
           last_intermed = last_intermed + 1

    if verbosity:
        print('vl_aib: parent of %d (fisrt) is now %d'%(first, parents[first]))

    cost  = cost - (nvlaues-1)
    for n in range(e, d-1, -1):
        cost[n] = cost[n-(e-ep)]


    for n in range(c, a-1, -1):
       cost[n] = cost[d]
       
    return parents, costs
    
