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


"""
Persistent Contrastive Divergence - Restricted Boltzmann Machine (PCD RBM)
( Stocastic Maximum Likelihood SML )

algo:
    set lr
    set k ( num of gibbs steps maybe 1-5 )
    initialize set of samples from a uniform distribution
    while not converged do
        sample from the training set
        g_pos = (1/m) * sum( delta_log_un_normalized_prob(normalized( x(i) )) )

        for i = 1 to k do 
            for j = 1 to m do 
                un_normalized(x(j) ) = gibbs_update( un_normalized(x(j)) )
        g_neg = (1/m) * sum( delta_log_un_normalized_prob(un_normalized(x(i))) )
        w = w + lr*(g_pos - g_neg)
"""


 References :

   - DeepLearningTutorials:   https://github.com/lisa-lab/DeepLearningTutorials

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

def sigmoid_fun(X):
    return 1./(1.+np.exp(-X))


class RBM():
	def __init__(self,num_hidden = 2, num_visible = 2):
		# initialize the weights & the hidden/visible units 
		self.hidden_units = num_hidden
		self.visible_units = num_visible
		self.b = np.zeros(self.visible_units) # initialize visible bias with 0s
		self.c = np.zeros(self.hidden_units) # initialize hidden bias with 0s
		# initialize weights uniformaly
		uniform_w = 1. / num_visible
		self.w = np.random.rand(num_visible,num_hidden) # use random array to do uniforming variables
		

	def train(self,lr = 0.01,k = 1,epochs=1000,training_data):
		self.lr = lr
		# set lr and k
		self.training_data = training_data
		# TODO: generate random samples from a uniform distribution
		'''
		TODO: change the sampling from the training set to initialize samples
		from a uniform distribution to follow the SML algo.
		'''

		# (repeat until converged ) loop for num of epochs
		for ep in range(epochs):
			''' +ve phase '''
			# sample from the training set
			_,h_new_samples = self.sample_h_given_v(self.training_data)
			# gibbs update by +ve gredient
			pos_G = np.dot(self.training_data.T, h_1st_sample)

			''' -ve phase '''
			# for steps in k-iterations:
			for i in range(k):
				# gibbs update
				v_samples,h_samples,prob_v,prob_h = self.gibbs_sample(h_new_samples)
			
			# gibbs update by -ve gredient
			neg_G = np.dot(v_samples.T, prob_h ) 
			# update the weights
			self.update_weights(pos_G,neg_G,h_new_samples,v_samples,prob_h)
		

	def gibbs_sample(self,old_h_samples):
		'''
		This function perform the gibbs sampling, Return a sampled version of the input.
		'''
		# update by calling v given h
		prob_v,v_samples=self.sample_v_given_h(old_h_samples)
		# then call h given v
		prob_h,h_samples = self.sample_h_given_v(v_samples)

		return v_samples,h_samples,prob_v,prob_h

	def update_weights(self,pos_G,neg_G,h_1st_sample,v_samples,prob_h):
		'''
		This function update the weights according to c-part
		w = w + lr(+ve Gredient - -ve Gredient)
		'''
				
		self.w += self.lr * (pos_G - neg_G)
		self.b += self.lr * np.mean(self.training_data - v_samples, axis=0)
        self.c += self.lr * np.mean(h_1st_sample - prob_h, axis=0)

	def sample_h_given_v(self,v_samples):
		'''
		sample hidden prob. given visible p(h|v)
		'''
		pre_sigmoid = np.dot(v_samples,self.w) + self.c
		prob_h = sigmoid_fun(pre_sigmoid)
		h_new_samples = np.random.binomial(size = self.hidden_units,n=1,p = prob_h) # calculate the new hidden samples
		return prob_h,h_new_samples


	def sample_v_given_h(self,h_samples):
		'''
		sample visible prob. given hidden p(v|h)
		'''
		pre_sigmoid = np.dot(h_samples,self.w) + self.b
		prob_v = sigmoid_fun(pre_sigmoid)
		v_new_samples = np.random.binomial(size = self.visible_units,n=1, p = prob_v) # calculate the new visible samples
		return prob_v,v_new_samples

	def test(self):
		# to test the rbm.
		pass

if __name__ == "__main__":
	training_data = load_X_data()
	rbm = RBM() # to set the hidden units and visible units
	rbm.train(training_data)# to set the learning rate and the k num
	rbm.test()