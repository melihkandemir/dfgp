from StructNoiseSparseGPRegressor import StructNoiseSparseGPRegressor
from RBFKernel import RBFKernel
from LinearKernel import LinearKernel
import numpy as np
import matplotlib.pyplot as plt

Ntr = 200
D = 1
Nts=200

Xtr = np.random.random([Ntr,D])-0.5 # N: 100, D: 20
ytr = np.where(Xtr<0.,-1.,1.)
Xts = np.random.random([Nts,D])-0.5
yts = np.where(Xts<0.,-1.,1.)

dof = 5 # how many hidden neurons to have
length_scale = 0.15
inducing_kernel = RBFKernel(length_scale)
kernels=list()  
for rr in range(dof):
   length_scale=np.sqrt(Xtr.shape[1])
   kernel=LinearKernel()
   kernels.append(kernel)
num_inducing=50
max_iter=5

model = StructNoiseSparseGPRegressor(inducing_kernel,kernels,num_inducing, max_iter, learnhyper=1, learn_sub_gps=1, learning_rate_start=0.00001)
model.train(Xtr,ytr)
predictions=model.predict(Xts)

yts=yts.ravel()

rmse=np.sqrt(np.mean((predictions-yts)*(predictions-yts)))

plt.close()
plt.plot(Xts.ravel(),predictions,'ro')
plt.show()


