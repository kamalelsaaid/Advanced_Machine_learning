import numpy as np



# generate the dataset

dt = np.dtype('>u4, >u4, >u4, >u4, (10000,784)u1')
mnist = np.fromfile('t10k-images-idx3-ubyte', dtype=dt)['f4'][0].T
imgs = np.zeros((784, 10000), dtype=np.dtype('b'))
imgs[mnist > 127] = 1