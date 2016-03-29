from rpca import RobustPCA
from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np


model = RobustPCA()

data = datasets.load_sample_images()
X = [np.mean(D, axis=-1) for D in data.images]
M = X[0] + np.random.laplace(scale=5, size=X[0].shape)
model.fit(M)
L = model.embedding_
S  = M - L


plt.figure()
plt.title('Noisy')
plt.imshow(M, cmap='gray')


plt.figure()
plt.title('Low-rank')
plt.imshow(L, cmap='gray')

print('Original stats (min=%f, max=%f)' % (np.min(M), np.max(M)))
print('low rank stats (min=%f, max=%f)' % (np.min(L), np.max(L)))

plt.show()