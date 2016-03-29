
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_mldata
from sklearn.metrics import roc_curve, auc
from numpy.linalg import norm
import numpy as np
from matplotlib import pyplot as plt
from pcp import pcp
from tga import TGA
from sklearn.utils import shuffle as sh
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from rpca import RobustPCA


def do_tga(M, n):
    tga = TGA(n_components=n, random_state=1, tol=1e-3)
    tga.fit(M)
    transformed = tga.transform(M)
    L = tga.inverse_transform(transformed)
    S = M - L
    return L, S, tga.obj, tga.nnzs


def do_pcp(M):
    L, S, obj, nnzs = pcp(M, verbose=True, delta=1e-6, mu=1, svd_method='exact', maxiter=100)
    return L, S, obj, nnzs


def do_rpca(M):
    model = RobustPCA(verbose=True, max_iter=100)
    L = model.fit_transform(M)
    S = M - L

    return L, S, model.diagnostics_['obj_list'], None


#dataset = fetch_covtype(shuffle=True)
datasets = ['SF']  # , 'smtp', 'SA', 'SF', 'shuttle', 'forestcover']
rank_matrix = {'SF': 3, 'SA': 25, 'shuttle': 5, 'forestcover': 17}

for dat in datasets:
    # loading and vectorization
    print('loading data')
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

print(dataset.data.shape)


new_X = np.zeros_like(X, dtype=np.int)

if (X.dtype == np.dtype('O')):
    print('X is object type')
    for c in range(X.shape[1]):
        new_X[:, c] = LabelEncoder().fit_transform(X[:, c])
    X = new_X

if y.dtype == np.dtype('O'):
    print('y is object type')
    y = LabelEncoder().fit_transform(y)

L, S, pcp_obj, _ = do_pcp(X)
print(pcp_obj)
L, S, rpca_obj, _ = do_rpca(X)

import os
import shutil
data = datasets[0]
shutil.rmtree(data, True)
os.mkdir(data)

f = plt.figure(figsize=(15, 10))
plt.plot(pcp_obj, label='Dan PCP Objective')

plt.plot(rpca_obj, label='Brian PCP Objective')
plt.legend()
plt.grid(True)
f.savefig(data + '/' + 'objective.png')

plt.show()
