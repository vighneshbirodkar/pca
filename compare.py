from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_mldata
import numpy as np
from numpy.linalg import matrix_rank
from matplotlib import pyplot as plt
from pcp import pcp
from tga import TGA
from sklearn.utils import shuffle as sh
from rpca import RobustPCA
from numpy.linalg import norm
from numpy.linalg import svd
from sklearn.preprocessing import LabelBinarizer
from mypca import mypcp


TOLERANCE = 1e-4


def prettify(ax):
    ax.legend(loc='best')
    ax.grid(True)


def do_tga(M, n):
    tga = TGA(n_components=n, random_state=1, tol=1e-3)
    tga.fit(M)
    transformed = tga.transform(M)
    L = tga.inverse_transform(transformed)
    S = M - L
    return L, S, tga.obj, tga.nnzs


def do_pcp(M, mu=None, report=False):

    if report:
        d_ = '-'*8
        print((d_ + 'Dan PCA for mu = %s' + d_) % mu)

    L, S, obj, nnzs = pcp(M, verbose=True, delta=1e-5, mu=mu,
                          svd_method='exact', maxiter=20)

    if report:
        lam = 1.0/np.sqrt(np.max(M.shape))
        nn = norm(L, 'nuc')
        on = np.sum(np.abs(S))
        o = nn + lam*on
        print('Rank = %d, NNZs = %d' % (matrix_rank(L),
                                        np.count_nonzero(S)))
        print('Nuclear Norm = %e' % nn)
        print('One Norm = %e' % on)
        print('Objective = %e' % o)

    return L, S, obj, nnzs


def do_mypcp(M, mu=None, lam=None, report=False, throttle=True):

    if report:
        d_ = '-'*8
        print((d_ + 'My PCA for mu = %s, throttle=%s' + d_) %
              (mu, str(throttle)))

    L, S, err_list = mypcp(M, mu=mu, lam=lam, max_iter=70, throttle=throttle)
    err_ax.semilogy(err_list, label='Throttle %s' % str(throttle))

    if report:
        lam = 1.0/np.sqrt(np.max(M.shape))
        nn = norm(L, 'nuc')
        on = np.sum(np.abs(S))
        o = nn + lam*on
        print('Rank = %d, NNZs = %d' % (matrix_rank(L),
                                        np.count_nonzero(S)))
        print('Nuclear Norm = %e' % nn)
        print('One Norm = %e' % on)
        print('Objective = %e' % o)

    return L, S


def do_rpca(M, report=False):

    if report:
        d_ = '-'*8
        print(d_ + 'Brian PCA' + d_)

    model = RobustPCA(verbose=False, max_iter=10, abs_tol=1e-10,
                      rel_tol=1e-7)
    L = model.fit_transform(M)
    S = M - L
    lam = 1.0/np.sqrt(np.max(M.shape))
    if report:
        nn = norm(L, 'nuc')
        on = np.sum(np.abs(S))
        o = nn + lam*on
        print('Rank = %d, NNZs = %d' % (matrix_rank(L),
                                        np.count_nonzero(S)))
        print('Nuclear Norm = %e' % nn)
        print('One Norm = %e' % on)
        print('Objective = %e' % o)

    fig, error_ax = plt.subplots()
    error_ax.semilogy(model.diagnostics_['err_primal'], label='Primal Error')
    error_ax.semilogy(model.diagnostics_['err_dual'], label='Dual Error')
    prettify(error_ax)
    return L, S, model.diagnostics_['objective_list'], None


#dataset = fetch_covtype(shuffle=True)
datasets = ['synthetic']  # , 'smtp', 'SA', 'SF', 'shuttle', 'forestcover']
rank_matrix = {'SF': 3, 'SA': 25, 'shuttle': 5, 'forestcover': 17}

fig, obj_ax = plt.subplots()
fig, svd_ax = plt.subplots()
fig, err_ax = plt.subplots()

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
        L0 = L
        S0 = S
        print('Data Rank = %f, Data NNZs = %f' %
              (matrix_rank(L, TOLERANCE), np.count_nonzero(S)))
        svd_ax.semilogy(svd(L, False, False), label='SVD Data')

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


for mu in []:
    L, S, pcp_obj, _ = do_pcp(X, mu, report=True)
    obj_ax.semilogy(pcp_obj, label='Dan Objective mu = %s' % mu)
    svd_ax.semilogy(svd(L, False, False), label='Dan SVD %s' % mu)

#L, S, rpca_obj, _ = do_rpca(X, report=True)
#obj_ax.semilogy(rpca_obj, label='Brian Objective')
#svd_ax.semilogy(svd(L, False, False), label='Brian SVD %s')

L, S = do_mypcp(X, report=True, throttle=True)
svd_ax.semilogy(svd(L, False, False), label='MyPCA SVD')

L, S = do_mypcp(X, report=True, throttle=False)
svd_ax.semilogy(svd(L, False, False), label='MyPCA SVD')


prettify(err_ax)
prettify(obj_ax)
prettify(svd_ax)
plt.show()
