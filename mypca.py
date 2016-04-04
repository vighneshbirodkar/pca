import numpy as np
from numpy.linalg import norm, matrix_rank
from scipy.linalg import svd, diagsvd


def soft_thresh(x, t):
    "Apply soft thresholding to an array"
    #print(x.shape, np.maximum(np.abs(x) - t, 0).shape)
    sign = np.sign(x)
    thresh = np.maximum(np.abs(x) - t, 0)
    assert sign.shape == thresh.shape
    return np.multiply(sign, thresh)


def norm_1(X):
    return np.sum(np.abs(X))


def sv_thresh(X, t):
    m, n = X.shape
    U, s, V = svd(X, full_matrices=False)
    s = soft_thresh(s, t)
    S = diagsvd(s, m, n)
    ret = np.dot(U, np.dot(S, V))
    assert ret.shape == X.shape
    return ret


def mypcp(M, lam=None, mu=None, max_iter=1000, sigma=1e-7, verbose=False,
          throttle=False):

    if lam is None:
        lam = 1.0/np.sqrt(max(M.shape))

    if mu is None:
        mu = np.prod(M.shape)/(4*norm_1(M))

    mu_ratio = 2.
    error_ratio = 10.

    print('Lambda = %f, mu = %f' % (lam, mu))
    S = np.zeros_like(M)
    Y = np.zeros_like(M)
    err_list = []
    
    for iter_ in range(max_iter):
        L = sv_thresh(M - S + (Y/mu), 1/mu)
        S_old = S
        S = soft_thresh(M - L + (Y/mu), lam/mu)
        Y = Y + mu*(M - L - S)

        primal_error = norm(M - L - S, 'fro')
        dual_error = norm(mu*(S - S_old), 'fro')

        if throttle:
            if primal_error > error_ratio*dual_error:
                mu = mu*mu_ratio
                #print('-------Inc mu------')
            elif dual_error > error_ratio*primal_error:
                mu = mu/mu_ratio
                #print('-------Dec mu------')

        err_ratio = norm(M - L - S, 'fro')/norm(M, 'fro')
        err_list.append(err_ratio)
        #print('Primal Error = %f, Dual Error = %f, Error Ratio = %f' %
        #      (primal_error, dual_error, err_ratio))
        if err_ratio < sigma:
            break

    #print('Iterations = %d' % iter_)
    if iter_ >= max_iter:
        if verbose:
            print('Max Iters Reached')

    return L, S, err_list
