'''
1- load the training set ( the images and the labels )
2- initialize the weights / hidden h & visible v units
3- generate random samples by gibbs sampler
4- 
	a- +ve phase
		sample from the dataset
		update hidden by the probability of hidden given visible p(h|v)
		calculate the gredient for the +ve phase
	b- -ve phase
		sample from the model
		repeat K times ( update hidden by the probability of hidden given visible p(h|v) )
		calculate the gredient for the -ve phase
	c- update the weights
		w = w + lr(+ve Gredient - -ve Gredient)
'''

import numpy as np

# generate the dataset
def load_data():
	dt = np.dtype('>u4, >u4, >u4, >u4, (10000,784)u1')
	mnist = np.fromfile('t10k-images-idx3-ubyte', dtype=dt)['f4'][0].T
	imgs = np.zeros((784, 10000), dtype=np.dtype('b'))
	imgs[mnist > 127] = 1

def sigmoid_fun(X):
    return 1./(1.+np.exp(-X))