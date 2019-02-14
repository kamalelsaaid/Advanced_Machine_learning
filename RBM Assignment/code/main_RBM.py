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
5- repeat 4

'''

import numpy as np

# generate the dataset
def load_X_data():
	'''
	To load the input training set images dataset
	'''
	dt = np.dtype('>u4, >u4, >u4, >u4, (10000,784)u1')
	mnist = np.fromfile('t10k-images-idx3-ubyte', dtype=dt)['f4'][0].T
	imgs = np.zeros((784, 10000), dtype=np.dtype('b'))
	imgs[mnist > 127] = 1

	return imgs

def load_Y_data():
	'''
	To load the labels of the dataset
	'''
	pass

def sigmoid_fun(X):
    return 1./(1.+np.exp(-X))


class RBM():
	def __init__(self,num_hidden = 2, num_visible = 2):
		# initialize the weights & the hidden/visible units 
		self.hidden_units = num_hidden
		self.visible_units = num_visible
		b_v = np.zeros(self.visible_units) # initialize visible bias with 0s
		c_h = np.zeros(self.hidden_units) # initialize hidden bias with 0s
		# initialize weights

	def gibbs_sample(self):
		'''
		This function perform the gibbs sampling
		'''
		pass

	def positive_phase(self):
		'''
		This function to perform the positive phase
		'''
		pass

	def negative_phase(self):
		'''
		This function to perform the negative phase
		'''
		pass

	def update_wrights(self):
		'''
		This function update the weights according to c-part
		w = w + lr(+ve Gredient - -ve Gredient)
		'''

	def sample_h_given_v(self):
		'''
		sample hidden prob. given visible p(h|v)
		'''
		pass

	def sample_v_given_h(self):
		'''
		sample visible prob. given hidden p(v|h)
		'''
		pass

if __name__ == "__main__":
	X = load_X_data()
	Y = load_Y_data()
	rbm = RBM()