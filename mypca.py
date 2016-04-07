import numpy as np
from numpy.linalg import norm, matrix_rank
from scipy.linalg import svd, diagsvd
from fbpca import pca
from sklearn.utils.extmath import randomized_svd


def soft_thresh(x, t):
    "Apply soft thresholding to an array"
    #print(x.shape, np.maximum(np.abs(x) - t, 0).shape)
    sign = np.sign(x)
    thresh = np.maximum(np.abs(x) - t, 0)
    assert sign.shape == thresh.shape
    return np.multiply(sign, thresh)


def norm_1(X):
    return np.sum(np.abs(X))


def sv_thresh(X, t, k):
    m, n = X.shape
    U, s, V = randomized_svd(X, k)  #pca(X, raw=True, k=25)
    # Number of singular values greater than `t`
    greater_sv = np.sum(s > t)
    s = soft_thresh(s, t)
    S = np.diag(s)
    ret = np.dot(U, np.dot(S, V))
    assert ret.shape == X.shape
    return ret, greater_sv


def mypcp(M, lam=None, mu=None, max_iter=1000, sigma=1e-7, verbose=False,
          throttle=False):
    # See http://arxiv.org/pdf/1009.5055v3.pdf

    if lam is None:
        lam = 1.0/np.sqrt(max(M.shape))

    eps_2 = 1e-2
    rho = 1.6
    d = min(M.shape)

    # See equation 10
    J = min(norm(M, 2), np.max(np.abs(M)))
    mu = 1.25/norm(M, 2)

    S = np.zeros_like(M)
    L = np.zeros_like(M)
    Y = M/J
    err_list = []
    M_norm = norm(M, 'fro')
    obj = []
    sv = 10
    for iter_ in range(max_iter):
        S = soft_thresh(M - L + (Y/mu), lam/mu)
        L, svp = sv_thresh(M - S + (Y/mu), 1/mu, sv)
        Y = Y + mu*(M - L - S)

        # Equation 25
        #if mu*delta_S/M_norm < eps_2:
        mu = rho*mu
        mu = min(mu, 1e4)
            #print('Mu updated to %e' % mu)

        if svp < sv:
            sv = svp + 1
        else:
            sv = svp + int(round(0.05*d))

        sv = min(sv, M.shape[0], M.shape[1])
        #print('sv = %d, svp = %d 1/mu = %f' % (sv, svp, 1/mu))

        o = norm(L, 'nuc') + lam*np.sum(np.abs(M - L))
        obj.append(o)
        err_ratio = norm(M - L - S, 'fro')/norm(M, 'fro')
        err_list.append(err_ratio)

        if err_ratio < sigma:
            break
        #mu = min(mu*1.1, 1e7)

    #print('Iterations = %d' % iter_)
    if iter_ >= max_iter:
        if verbose:
            print('Max Iters Reached')

    return L, S, obj, err_list
