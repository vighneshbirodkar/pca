
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_mldata
import numpy as np
from numpy.linalg import matrix_rank
from matplotlib import pyplot as plt
from pcp import pcp
from tga import TGA
from sklearn.utils import shuffle as sh
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
from rpca import RobustPCA
from numpy.linalg import norm


def do_tga(M, n):
    tga = TGA(n_components=n, random_state=1, tol=1e-3)
    tga.fit(M)
    transformed = tga.transform(M)
    L = tga.inverse_transform(transformed)
    S = M - L
    return L, S, tga.obj, tga.nnzs


def do_pcp(M, mu=None):
    L, S, obj, nnzs = pcp(M, verbose=True, delta=1e-6, mu=mu, svd_method='exact', maxiter=1000)
    return L, S, obj, nnzs


def do_rpca(M):
    model = RobustPCA(verbose=False, max_iter=1000)
    L = model.fit_transform(M)
    S = M - L

    np.save('err_primal.npy', model.diagnostics_['err_primal'])
    np.save('err_dual.npy', model.diagnostics_['err_dual'])
    return L, S, model.diagnostics_['obj_list'], None


#dataset = fetch_covtype(shuffle=True)
datasets = ['synthetic']  # , 'smtp', 'SA', 'SF', 'shuttle', 'forestcover']
rank_matrix = {'SF': 3, 'SA': 25, 'shuttle': 5, 'forestcover': 17}

for dat in datasets:
    # loading and vectorization
    print('loading data')
    if dat == 'synthetic':
        n = 500
        r = 25  # Rank
        X = np.random.normal(0, 1/float(n), size=(n, r))
        Y = np.random.normal(0, 1/float(n), size=(n, r))
        L = np.dot(X, Y.T)
        S = np.random.choice([0, 1, -1], (n, n), p=[0.95, 0.025, 0.025])
        X = L + S

    if dat in ['http', 'smtp', 'SA', 'SF']:
        dataset = fetch_kddcup99(subset=dat, shuffle=True, percent10=True)
        X = dataset.data
        y = dataset.target

    if dat == 'shuttle':
        dataset = fetch_mldata('shuttle')
        X = dataset.data
        y = dataset.target
        sh(X, y)
        # we remove data with label 4
        # normal data are then those of class 1
        s = (y != 4)
        X = X[s, :]
        y = y[s]
        y = (y != 1).astype(int)

    if dat == 'forestcover':
        dataset = fetch_covtype(shuffle=True)
        X = dataset.data
        y = dataset.target
        # normal data are those with attribute 2
        # abnormal those with attribute 4
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        y = (y != 2).astype(int)

    print('vectorizing data')

    if dat == 'SF':
        lb = LabelBinarizer()
        lb.fit(X[:, 1])
        x1 = lb.transform(X[:, 1])
        X = np.c_[X[:, :1], x1, X[:, 2:]]
        y = (y != 'normal.').astype(int)

    if dat == 'SA':
        lb = LabelBinarizer()
        lb.fit(X[:, 1])
        x1 = lb.transform(X[:, 1])
        lb.fit(X[:, 2])
        x2 = lb.transform(X[:, 2])
        lb.fit(X[:, 3])
        x3 = lb.transform(X[:, 3])
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        y = (y != 'normal.').astype(int)

#print(dataset.data.shape)

new_X = np.zeros_like(X, dtype=np.int)

if (X.dtype == np.dtype('O')):
    print('X is object type')
    for c in range(X.shape[1]):
        new_X[:, c] = LabelEncoder().fit_transform(X[:, c])
    X = new_X

#if y.dtype == np.dtype('O'):
#    print('y is object type')
#    y = LabelEncoder().fit_transform(y)

print(X.shape)

X = MinMaxScaler().fit_transform(X)
plt.figure(figsize=(15, 10))
lam = 1.0 / np.sqrt(np.max(X.shape))
d_ = '-'*10

for mu in [None, 0.1, 10]:
    print((d_ + 'Dan PCA for mu = %s' + d_) % mu)
    L, S, pcp_obj, _ = do_pcp(X, mu)
    nn = norm(L, 'nuc')
    on = np.sum(np.abs(S))
    o = nn + lam*on

    plt.plot(pcp_obj, label='Dan PCP Objective (mu = %s)' % mu)
    print('Rank = %d, NNZs = %d' % (matrix_rank(L), np.count_nonzero(S)))
    print('Nuclear Norm = %e' % nn)
    print('One Norm = %e' % on)
    print('Objective = %e' % o)
    np.save('dan_obj_' + str(mu) + '.npy', np.array(pcp_obj))
    np.save('dan_L_' + str(mu) + '.npy', L)

L, S, rpca_obj, _ = do_rpca(X)
print(d_ + 'Brian PCA' + d_)
nn = norm(L, 'nuc')
on = np.sum(np.abs(S))
o = nn + lam*on
print('Rank = %d, NNZs = %d' % (matrix_rank(L), np.count_nonzero(S)))
print('Nuclear Norm = %e' % nn)
print('One Norm = %e' % on)
print('Objective = %e' % o)
np.save('brian_L.npy', L)
np.save('brian_obj.npy', np.array(rpca_obj))

plt.plot(rpca_obj, label='Brian PCP Objective')

#import os
#import shutil
#data = datasets[0]
#shutil.rmtree(data, True)
#os.mkdir(data)


plt.legend(loc='best')
plt.grid(True)
#f.savefig(data + '/' + 'objective.png')

plt.show()
