from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_mldata
import numpy as np
from numpy.linalg import matrix_rank
from matplotlib import pyplot as plt
from pcp import pcp
from sklearn.utils import shuffle as sh
from rpca import RobustPCA
from numpy.linalg import norm
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from mypca import mypcp


def prettify(ax):
    ax.legend(loc='best')
    ax.grid(True)


def gen_synthetic(n, sparseness, rank):
    r = rank  # Rank
    X = np.random.normal(0, 1/float(n), size=(n, r))
    Y = np.random.normal(0, 1/float(n), size=(n, r))
    L = np.dot(X, Y.T)
    p = sparseness/2
    S = np.random.choice([0, 1, -1], size=(n, n), p=[1 - 2*p, p, p])

    return L, S


def gen_report(M, name, obj, err, L_test, S_test, L_true, S_true,
               y_true):

    lam = 1.0/np.sqrt(np.max(M.shape))
    nn = norm(L_test, 'nuc')
    on = np.sum(np.abs(S_test))
    o = nn + lam*on
    print('Rank = %d, NNZs = %d' % (matrix_rank(L_test),
                                    np.count_nonzero(S_test)))
    print('Nuclear Norm = %e' % nn)
    print('One Norm = %e' % on)
    print('Objective = %e' % o)
    if L_true is not None:
        print('Recovery Error = %e' %
              (norm(L_test - L_true, 'fro')/norm(L_true, 'fro')))

    y_test = np.linalg.norm(S_test, axis=1)
    tp, fp, _ = metrics.roc_curve(y_true, y_test)
    score = metrics.roc_auc_score(y_true, y_test)
    auc_ax.plot(tp, fp, label=name + ' AUC=' + str(score))
    obj_ax.plot(obj, label=name + ' Objective')


def do_pcp(M=None, y_true=None, L=None, S=None, mu=None,
           lam=None, report=False):

    if M is None:
        M = L + S

    L_true, S_true = L, S
    if report:
        d_ = '-'*8
        print((d_ + 'Dan PCA for mu = %s' + d_) % mu)

    L, S, obj, err = pcp(M, verbose=True,
                         svd_method='exact', maxiter=100)

    if report:
        gen_report(M, 'Dan PCP', obj, err, L_true=L_true, S_true=S_true,
                   L_test=L, S_test=S, y_true=y_true)
    return L, S, obj


def do_mypcp(M=None, y_true=None, L=None, S=None, mu=None,
             lam=None, report=False):

    if M is None:
        M = L + S

    L_true, S_true = L, S
    if report:
        d_ = '-'*8
        print(d_ + 'My PCA' + d_)

    L, S, obj_list, err_list = mypcp(M, max_iter=100)

    if report:
        gen_report(M, 'My PCP', obj_list, err_list,
                   L_true=L_true, S_true=S_true, y_true=y_true,
                   L_test=L, S_test=S)
    return L, S


def do_rpca(M=None, y_true=None, L=None, S=None, mu=None,
            lam=None, report=False):

    if L is not None:
        M = L + S

    L_true, S_true = L, S
    if report:
        d_ = '-'*8
        print(d_ + 'Brian PCA' + d_)

    model = RobustPCA(verbose=False, max_iter=100)
    L = model.fit_transform(M)
    S = M - L
    if report:
        gen_report(M, 'Brian PCP', model.diagnostics_['objective_list'],
                   None, L_true=L_true, S_true=S_true, L_test=L, S_test=S,
                   y_true=y_true)
    return L, S, model.diagnostics_['objective_list'], None


#dataset = fetch_covtype(shuffle=True)
datasets = ['forestcover']  # , 'smtp', 'SA', 'SF', 'shuttle', 'forestcover']
rank_matrix = {'SF': 3, 'SA': 25, 'shuttle': 5, 'forestcover': 17}

fig, obj_ax = plt.subplots()
fig, auc_ax = plt.subplots()
#fig, svd_ax = plt.subplots()
#fig, err_ax = plt.subplots()

for dat in datasets:
    # loading and vectorization
    if dat == 'synthetic1':

        L, S = gen_synthetic(500, 0.05, 25)
        X = L + S
        print('Data Rank = %d, Data NNZs = %d' %
              (matrix_rank(L), np.count_nonzero(S)))

    if dat == 'synthetic2':

        L, S = gen_synthetic(1000, 0.05, 25)
        X = L + S
        print('Data Rank = %d, Data NNZs = %d' %
              (matrix_rank(L), np.count_nonzero(S)))

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

#X = (X - X.min())/(X.max() - X.min())
if (X.dtype == np.dtype('O')):
    new_X = np.zeros_like(X, dtype=np.int)
    for c in range(X.shape[1]):
        new_X[:, c] = LabelEncoder().fit_transform(X[:, c])
    X = new_X

do_pcp(X, y_true=y, report=True)
do_rpca(X, y_true=y, report=True)
do_mypcp(X, y_true=y, report=True)

prettify(obj_ax)
prettify(auc_ax)

plt.show()
