import numpy as np
import scipy.linalg
import scipy.weave
from sklearn.base import BaseEstimator
from numpy.linalg import norm


class RobustPCA(BaseEstimator):
    '''Robust PCA

    http://arxiv.org/abs/0912.3599
    '''

    def __nuclear_prox(self, A, r=1.0):
        '''Proximal operator for scaled nuclear norm:
        Y* <- argmin_Y  r * ||Y||_* + 1/2 * ||Y - A||_F^2

        Arguments:
            A    -- (ndarray) input matrix
            r    -- (float>0) scaling factor

        Returns:
            Y    -- (ndarray) if A = USV', then Y = UTV'
                              where T = max(S - r, 0)
        '''

        U, S, V = scipy.linalg.svd(A, full_matrices=False)

        T = np.maximum(S - r, 0.0)

        Y = (U * T).dot(V)
        return Y

    def __l1_prox(self, A, r=1.0):
        '''Proximal operator for entry-wise matrix l1 norm:
        Y* <- argmin_Y r * ||Y||_1 + 1/2 * ||Y - A||_F^2

        Arguments:
            A    -- (ndarray) input matrix
            r    -- (float>0) scaling factor

        Returns:
            Y    -- (ndarray) Y = A after shrinkage
        '''

        Y = np.zeros_like(A)
        numel = A.size

        shrinkage = r"""
            for (int i = 0; i < numel; i++) {
                Y[i] = 0;
                if (A[i] - r > 0) {
                    Y[i] = A[i] - r;
                } else if (A[i] + r <= 0) {
                    Y[i] = A[i] + r;
                }
            }
        """
        scipy.weave.inline(shrinkage, ['numel', 'A', 'r', 'Y'])
        return Y

    def __cost(self, Y, Z):
        '''Get the cost of an RPCA solution.

        Arguments:
            Y       -- (ndarray)    the low-rank component
            Z       -- (ndarray)    the sparse component
            alpha   -- (float>0)    the balancing factor

        Returns:
            total, nuclear_norm, l1_norm -- (list of floats)
        '''
        nuclear_norm = scipy.linalg.svd(Y,
                                        full_matrices=False,
                                        compute_uv=False).sum()

        l1_norm = np.abs(Z).sum()
        #print(l1_norm, norm_(Z, 1))

        return nuclear_norm + self.alpha_ * l1_norm, nuclear_norm, l1_norm

    def __init__(self, alpha=None, max_iter=200, verbose=False, abs_tol=1e-4,
                 rel_tol=1e-3):
        '''
        Arguments:
            alpha -- (float > 0) weight between low-rank and noise term
                     If left as None, alpha will be automatically set to
                     sqrt(max(X.shape))
            max_iter -- (int > 0) maximum number of iterations
        '''

        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    def fit(self, X):
        '''Fit the robust PCA model to a matrix X'''

        self.fit_transform(X)
        return self

    def fit_transform(self, X):

        # Some magic numbers for dynamic augmenting penalties in ADMM.
        # Changing these shouldn't effect correctness, only convergence rate.

        RHO_START = np.prod(X.shape) / (4*np.abs(X).sum())
        RHO_MIN = -np.inf
        RHO_MAX = np.inf
        MAX_RATIO = 2e0
        SCALE_FACTOR = 1.5e0

        ABS_TOL = self.abs_tol
        REL_TOL = self.rel_tol

        # update rules:
        #  Y+ <- nuclear_prox(X - Z - W, 1/rho)
        #  Z+ <- l1_prox(X - Y - W, alpha/rho)
        #  W+ <- W + Y + Z - X

        # Initialize
        rho = RHO_START

        # Scale the data to a workable range
        X = X.astype(np.float)
        Xmin = np.min(X)
        rescale = max(1e-8, np.max(X - Xmin))

        Xt = (X - Xmin) / rescale

        Y = Xt.copy()
        Z = np.zeros_like(Xt)
        W = np.zeros_like(Xt)

        norm_X = scipy.linalg.norm(Xt)

        if self.alpha is None:
            self.alpha_ = max(Xt.shape)**(-0.5)
        else:
            self.alpha_ = self.alpha

        m = X.size

        _DIAG = {
            'err_primal': [],
            'err_dual':   [],
            'eps_primal': [],
            'eps_dual':   [],
            'rho':        [],
            'objective_list':   []
        }

        # For Boyd ADMM paper
        # Xt -> c
        # Z -> z (sparse)
        # Y -> x (low rank)
        # W -> u (lagrange multiplier)
        for t in range(self.max_iter):
            # Eq 3.5
            Y = self.__nuclear_prox(Xt - Z - W, 1.0/rho)
            Z_old = Z.copy()
            # Step 3.6
            Z = self.__l1_prox(Xt - Y - W, self.alpha_ / rho)

            residual_pri = Y + Z - Xt
            residual_dual = Z - Z_old

            res_norm_pri = scipy.linalg.norm(residual_pri)
            res_norm_dual = rho * scipy.linalg.norm(residual_dual)

            #Step 3.7
            W = W + residual_pri

            # Just after 3.12
            max_norm = max(scipy.linalg.norm(Y), scipy.linalg.norm(Z), norm_X)
            eps_pri = np.sqrt(m) * ABS_TOL + REL_TOL * max_norm
            eps_dual = np.sqrt(m) * ABS_TOL + REL_TOL * scipy.linalg.norm(W)

            _DIAG['eps_primal'].append(eps_pri)
            _DIAG['eps_dual'].append(eps_dual)
            _DIAG['err_primal'].append(res_norm_pri)
            _DIAG['err_dual'].append(res_norm_dual)
            _DIAG['rho'].append(rho)

            if res_norm_pri <= eps_pri and res_norm_dual <= eps_dual:
                break

            # Equation 3.13
            if (res_norm_pri > MAX_RATIO*res_norm_dual and
                    rho*SCALE_FACTOR <= RHO_MAX):
                rho = rho * SCALE_FACTOR
                W = W / SCALE_FACTOR

            elif (res_norm_dual > MAX_RATIO * res_norm_pri and
                  rho / SCALE_FACTOR >= RHO_MIN):
                rho = rho / SCALE_FACTOR
                W = W * SCALE_FACTOR

            Z2 = (Z) * rescale + Xmin
            L = X - Z2
            obj = self.__cost(L, Z2)[0]
            _DIAG['objective_list'].append(obj)

        if self.verbose:
            if t < self.max_iter - 1:
                print 'Converged in %d steps' % t
            else:
                print 'Reached maximum iterations'

        # Scale back up to the original data scale
        Z = Z*rescale + Xmin
        self.embedding_ = X - Z

        _DIAG['cost'] = self.__cost(self.embedding_, Z)

        self.diagnostics_ = _DIAG

        return self.embedding_
