import vlfeat
import pdb
import numpy as np
import cPickle as pickle
import matplotlib
from vlfeat.kmeans.hikmeans_print import hikmeans_print, hik_py_print
if __name__ == '__main__':
    path_data = '/home/sh/tmp/hikmeans/data.pkl'
    #data  = np.random.rand(2,10000)*255
    #data = np.asarray(data, dtype='uint8')
    #pickle.dump(data, open(path_data,'wb'))

    
    #pdb.set_trace()
    data = pickle.load(open(path_data, 'rb'))
    #pdb.set_trace()


    
    K = 3
    nleaves = 100
    tree, asgn1, hik1 = vlfeat.vl_hikmeans(data, K, nleaves, verbosity=1, method='elkan')

    #hikmeans_print(hik1)
    #hik_py_print(tree)
    #pdb.set_trace()
    
    #path = '/home/sh/tmp/hikmeans/tree.pkl'
    #pickle.dump(tree,open(path,'wb'))

    #tree2 = pickle.load(open(path,'rb'))
    #datat = np.random.rand(255,100000)*255
    asgn2, hik2 = vlfeat.vl_hikmeanspush(tree, data, verbosity=1, method='lloyd')

    pdb.set_trace()

    
    hist = vlfeat.vl_hikmeanshist(tree2, asgn2)
    
    pdb.set_trace()


