import vlfeat
import pdb
import numpy as np
import cPickle as pickle
import matplotlib
from vlfeat.kmeans.hikmeans_print import hikmeans_print, hik_py_print

import scipy.io as scio
if __name__ == '__main__':
    path_data = '/home/sh/tmp/hikmeans/data.mat'
    #data  = np.random.rand(2,10000)*255
    #data = np.asarray(data, dtype='uint8')
    #pickle.dump(data, open(path_data,'wb'))

    
    #pdb.set_trace()
    data = scio.loadmat(path_data)['data']
    #pdb.set_trace()


    
    K = 3
    nleaves = 100
    tree, asgn1,h1 = vlfeat.vl_hikmeans(data, K, nleaves)

    pdb.set_trace()
    
    #hikmeans_print(hik1)
    #hik_py_print(tree)
    #pdb.set_trace()
    
    #path = '/home/sh/tmp/hikmeans/tree.pkl'
    #pickle.dump(tree,open(path,'wb'))

    #tree2 = pickle.load(open(path,'rb'))
    
    #print(str(asgn1))
    for i in range(1):
        asgn2,h2 = vlfeat.vl_hikmeanspush(tree, data)
        #print(str(asgn2))
        #print((asgn1==asgn2))

    #path_asgn =  '/home/sh/tmp/hikmeans/asgn.mat'
    #asgn2 = scio.loadmat(path_asgn)['A']
    #hist = vlfeat.vl_hikmeanshist(tree2, asgn2)
    
    pdb.set_trace()


