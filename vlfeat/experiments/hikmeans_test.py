import vlfeat
import pdb
import numpy as np
import cPickle as pickle
if __name__ == '__main__':
    
    data  = np.random.rand(2,10000)*255
    data = np.asarray(data, dtype='uint8')
    K = 3
    nleaves = 100
    hikmeans, tree = vlfeat.vl_hikmeans(data, K, nleaves, verbose=1)

    pdb.set_trace()
    
    datat = np.random.rand(2,100000)*255
    AT, hikmeans_2 = vlfeat.vl_hikmeanspush(tree, datat)

    print (hikmeans == hikmeans_2)

    #path = '/home/sh/tmp/hikmeans'
    #pickle.dump(tree, open(path, 'wb'))
    #tree_load = pickle.load(open(path, 'rb'))
    #AT_load = vlfeat.vl_hikmeanspush(tree_load, datat)


    
    pdb.set_trace()
