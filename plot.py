import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import svd


plt.figure()
L = np.load('dan_L_None.npy')
v = svd(L, False, False)
plt.plot(v, label='Dan SVD mu=None')

L = np.load('dan_L_0.01.npy')
v = svd(L, False, False)
plt.plot(v, label='Dan SVD mu=0.01')

L = np.load('brian_L.npy')
v = svd(L, False, False)
plt.plot(v, label='Brian SVD')

plt.legend()
plt.yscale('log')

plt.figure()
plt.plot(np.load('dan_obj_None.npy'), label='Dan Objective None')
plt.plot(np.load('dan_obj_0.01.npy'), label='Dan Objective 0.01')
plt.plot(np.load('brian_obj.npy'), label='Brian Objective')
plt.legend()

plt.figure()
plt.plot(np.load('err_primal.npy'), label='Primal Error')
plt.plot(np.load('err_dual.npy'), label='Dual Error')
plt.legend()
plt.show()
