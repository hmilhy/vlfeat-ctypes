import vlfeat
import pdb
import numpy as np
import cPickle as pickle
if __name__ == '__main__':
    
    data  = np.random.rand(2,10000)*255
    data = np.asarray(data, dtype='uint8')
    K = 3
    nleaves = 100
    tree = vlfeat.vl_hikmeans(data, K, nleaves, verbose=1)

    path = '/home/sh/tmp/hikmeans/tree.pkl'
    pickle.dump(tree,open(path,'wb'))

    tree2 = pickle.load(open(path,'rb'))
    datat = np.random.rand(2,100000)*255
    AT = vlfeat.vl_hikmeanspush(tree2, datat)

    pdb.set_trace()


