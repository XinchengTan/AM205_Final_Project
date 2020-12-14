from collections.abc import Sequence
import warnings
import operator
import sys
import time
from RLA import *

import numpy as np
from scipy import linalg
from scipy.linalg import fractional_matrix_power

from sklearn.covariance import empirical_covariance, EmpiricalCovariance, log_likelihood

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_random_state, check_array
from sklearn.linear_model import lars_path_gram

# TODO: all coord_descent implementations used _cd_fast's enet_coordinate_descent_gram
# from sklearn.linear_model import coordinate_descent, _coordinate_descent, cd_fast
from enet_coord_descent_gram import enet_coordinate_descent_gram


from sklearn.model_selection import check_cv, cross_val_score


# Helper functions to compute the objective and dual objective functions
# of the l1-penalized estimator
def _objective(mle, precision_, alpha):
    """Evaluation of the graphical-lasso objective function

    the objective function is made of a shifted scaled version of the
    normalized log-likelihood (i.e. its empirical mean over the samples) and a
    penalisation term to promote sparsity
    """
    p = precision_.shape[0]
    cost = - 2. * log_likelihood(mle, precision_) + p * np.log(2 * np.pi)
    cost += alpha * (np.abs(precision_).sum()
                     - np.abs(np.diag(precision_)).sum())
    return cost


def _dual_gap(emp_cov, precision_, alpha):
    """Expression of the dual gap convergence criterion

    The specific definition is given in Duchi "Projected Subgradient Methods
    for Learning Sparse Gaussians".
    """
    gap = np.sum(emp_cov * precision_)
    gap -= precision_.shape[0]
    gap += alpha * (np.abs(precision_).sum()
                    - np.abs(np.diag(precision_)).sum())
    return gap


def alpha_max(emp_cov):
    """Find the maximum alpha for which there are some non-zeros off-diagonal.

    Parameters
    ----------
    emp_cov : 2D array, (n_features, n_features)
        The sample covariance matrix

    Notes
    -----

    This results from the bound for the all the Lasso that are solved
    in GraphicalLasso: each time, the row of cov corresponds to Xy. As the
    bound for alpha is given by `max(abs(Xy))`, the result follows.

    """
    A = np.copy(emp_cov)
    A.flat[::A.shape[0] + 1] = 0
    return np.max(np.abs(A))


# The g-lasso algorithm

def graphical_lasso(emp_cov, alpha, cov_init=None, mode='cd', tol=1e-4,
                    enet_tol=1e-4, max_iter=100, verbose=False,
                    return_costs=False, eps=np.finfo(np.float64).eps,
                    return_n_iter=False, isRLA=False):
    """l1-penalized covariance estimator

    Read more in the :ref:`User Guide <sparse_inverse_covariance>`.

    Parameters
    ----------
    emp_cov : 2D ndarray, shape (n_features, n_features)
        Empirical covariance from which to compute the covariance estimate.

    alpha : positive float
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.

    cov_init : 2D array (n_features, n_features), optional
        The initial guess for the covariance.

    mode : {'cd', 'lars'}
        The Lasso solver to use: coordinate descent or LARS. Use LARS for
        very sparse underlying graphs, where p > n. Elsewhere prefer cd
        which is more numerically stable.

    tol : positive float, optional
        The tolerance to declare convergence: if the dual gap goes below
        this value, iterations are stopped.

    enet_tol : positive float, optional
        The tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'.

    max_iter : integer, optional
        The maximum number of iterations.

    verbose : boolean, optional
        If verbose is True, the objective function and dual gap are
        printed at each iteration.

    return_costs : boolean, optional
        If return_costs is True, the objective function and dual gap
        at each iteration are returned.

    eps : float, optional
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems.

    return_n_iter : bool, optional
        Whether or not to return the number of iterations.

    Returns
    -------
    covariance : 2D ndarray, shape (n_features, n_features)
        The estimated covariance matrix.

    precision : 2D ndarray, shape (n_features, n_features)
        The estimated (sparse) precision matrix.

    costs : list of (objective, dual_gap) pairs
        The list of values of the objective function and the dual gap at
        each iteration. Returned only if return_costs is True.

    n_iter : int
        Number of iterations. Returned only if `return_n_iter` is set to True.

    See Also
    --------
    GraphicalLasso, GraphicalLassoCV

    Notes
    -----
    The algorithm employed to solve this problem is the GLasso algorithm,
    from the Friedman 2008 Biostatistics paper. It is the same algorithm
    as in the R `glasso` package.

    One possible difference with the `glasso` R package is that the
    diagonal coefficients are not penalized.

    """
    _, n_features = emp_cov.shape
    if alpha == 0:
        if return_costs:
            precision_ = linalg.inv(emp_cov)
            cost = - 2. * log_likelihood(emp_cov, precision_)
            cost += n_features * np.log(2 * np.pi)
            d_gap = np.sum(emp_cov * precision_) - n_features
            if return_n_iter:
                return emp_cov, precision_, (cost, d_gap), 0
            else:
                return emp_cov, precision_, (cost, d_gap)
        else:
            if return_n_iter:
                return emp_cov, linalg.inv(emp_cov), 0
            else:
                return emp_cov, linalg.inv(emp_cov)
    if cov_init is None:
        covariance_ = emp_cov.copy()
    else:
        covariance_ = cov_init.copy()
    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends to
    # make the convergence go faster.
    covariance_ *= 0.95
    diagonal = emp_cov.flat[::n_features + 1]
    covariance_.flat[::n_features + 1] = diagonal
    precision_ = linalg.pinvh(covariance_)

    indices = np.arange(n_features)
    costs = list()
    # The different l1 regression solver have different numerical errors
    if mode == 'cd':
        errors = dict(over='raise', invalid='ignore')
    else:
        errors = dict(invalid='raise')
    try:
        # be robust to the max_iter=0 edge case, see:
        # https://github.com/scikit-learn/scikit-learn/issues/4134
        d_gap = np.inf
        # set a sub_covariance buffer
        sub_covariance = np.copy(covariance_[1:, 1:], order='C')
        for i in range(max_iter):
            for idx in range(n_features):
                # To keep the contiguous matrix `sub_covariance` equal to
                # covariance_[indices != idx].T[indices != idx]
                # we only need to update 1 column and 1 line when idx changes
                if idx > 0:
                    di = idx - 1
                    sub_covariance[di] = covariance_[di][indices != idx]
                    sub_covariance[:, di] = covariance_[:, di][indices != idx]
                else:
                    sub_covariance[:] = covariance_[1:, 1:]
                row = emp_cov[idx, indices != idx]
                with np.errstate(**errors):
                    if mode == 'cd':
                        # Use coordinate descent
                        coefs = -(precision_[indices != idx, idx]
                                  / (precision_[idx, idx] + 1000 * eps))
                        # TODO: swap in RLA LASSO with the following enet_coordinate_descent_gram() function!
                        if isRLA:
                            coefs = SLRviaRP(sub_covariance, 
                                             np.matmul(fractional_matrix_power(sub_covariance, -0.5), row),
                                             0.01, 0.05, 30, 1.1, 100, gamma=0)
                        else:
                            coefs, _, _, _ = enet_coordinate_descent_gram(
                                coefs, alpha, 0, sub_covariance,
                                row, row, max_iter, enet_tol,
                                check_random_state(None), False)
                    else:
                        # Use LARS
                        _, _, coefs = lars_path_gram(
                            Xy=row, Gram=sub_covariance, n_samples=row.size,
                            alpha_min=alpha / (n_features - 1), copy_Gram=True,
                            eps=eps, method='lars', return_path=False)
                # Update the precision matrix
                precision_[idx, idx] = (
                    1. / (covariance_[idx, idx]
                          - np.dot(covariance_[indices != idx, idx], coefs)))
                precision_[indices != idx, idx] = (- precision_[idx, idx]
                                                   * coefs)
                precision_[idx, indices != idx] = (- precision_[idx, idx]
                                                   * coefs)
                coefs = np.dot(sub_covariance, coefs)
                covariance_[idx, indices != idx] = coefs
                covariance_[indices != idx, idx] = coefs
            if not np.isfinite(precision_.sum()):
                raise FloatingPointError('The system is too ill-conditioned '
                                         'for this solver')
            d_gap = _dual_gap(emp_cov, precision_, alpha)
            cost = _objective(emp_cov, precision_, alpha)
            if verbose:
                print('[graphical_lasso] Iteration '
                      '% 3i, cost % 3.2e, dual gap %.3e'
                      % (i, cost, d_gap))
            if return_costs:
                costs.append((cost, d_gap))
            if np.abs(d_gap) < tol:
                break
            if not np.isfinite(cost) and i > 0:
                raise FloatingPointError('Non SPD result: the system is '
                                         'too ill-conditioned for this solver')
        else:
            warnings.warn('graphical_lasso: did not converge after '
                          '%i iteration: dual gap: %.3e'
                          % (max_iter, d_gap), ConvergenceWarning)
    except FloatingPointError as e:
        e.args = (e.args[0]
                  + '. The system is too ill-conditioned for this solver',)
        raise e

    if return_costs:
        if return_n_iter:
            return covariance_, precision_, costs, i + 1
        else:
            return covariance_, precision_, costs
    else:
        if return_n_iter:
            return covariance_, precision_, i + 1
        else:
            return covariance_, precision_

