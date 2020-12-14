
import numpy as np
# from numpy.random import randint
from sklearn.exceptions import ConvergenceWarning

RAND_R_MAX = 2147483647

RNG = np.random.mtrand._rand  # Random state singleton




fmax = lambda x, y: x if x > y else y


# def abs_max(n, a):  # floating* a
#     """np.max(np.abs(a))"""
#     m = fabs(a[0])
#     for i in range(1, n):
#         d = fabs(a[i])
#         if d > m:
#             m = d
#     return m


# def _asum(n, x, incx):  # floating * x
#   """sum(|x_i|)"""
#
#   # if floating is float:
#   #   return sasum( & n, x, & incx)
#   #   else:
#   #   return dasum( & n, x, & incx)
#   return


def enet_coordinate_descent_gram(w,
                                 alpha, beta,
                                 Q,
                                 q,
                                 y,
                                 max_iter, tol, rng=RNG,
                                 random=0, positive=0):
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression
        We minimize
        (1/2) * w^T Q w - q^T w + alpha norm(w, 1) + (beta/2) * norm(w, 2)^2
        which amount to the Elastic-Net problem when:
        Q = X^T X (Gram matrix)
        q = X^T y
    """
    dtype = np.array(w).dtype
    if np.issubdtype(np.float32, dtype):
      dtype = np.float64

    # get the data information into easy vars
    n_samples = y.shape[0]
    n_features = Q.shape[0]

    # initial value "Q w" which will be kept of up to date in the iterations
    H = np.dot(Q, w)


    XtA = np.zeros(n_features, dtype=dtype)
    # cdef floating tmp
    # cdef floating w_ii
    # cdef floating d_w_max
    # cdef floating w_max
    # cdef floating d_w_ii
    # cdef floating q_dot_w
    # cdef floating w_norm2
    gap = tol + 1.0
    d_w_tol = tol
    #cdef floating dual_norm_XtA
    #cdef unsigned int ii
    n_iter = 0
    #cdef unsigned int f_iter
    rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    #cdef UINT32_t* rand_r_state = &rand_r_state_seed

    y_norm2 = np.dot(y, y)
    # cdef floating* w_ptr = <floating*>&w[0]
    # cdef floating* Q_ptr = &Q[0, 0]
    # cdef floating* q_ptr = <floating*>q.data
    # cdef floating* H_ptr = &H[0]
    # cdef floating* XtA_ptr = &XtA[0]
    tol = tol * y_norm2

    if alpha == 0:
        raise Warning("Coordinate descent with alpha=0 may lead to unexpected"
            " results and is discouraged.")

    for n_iter in range(max_iter):
        w_max = 0.0
        d_w_max = 0.0
        for f_iter in range(n_features):  # Loop over coordinates
            if random:
                ii = np.random.randint(0, n_features)
                #ii = rand_int(n_features, rand_r_state)
            else:
                ii = f_iter

            if Q[ii, ii] == 0.0:
                continue

            w_ii = w[ii]  # Store previous value

            if w_ii != 0.0:
                # H -= w_ii * Q[ii]
                H -= w_ii * Q[ii]
                # _axpy(n_features, -w_ii, Q_ptr + ii * n_features, 1, H_ptr, 1)

            tmp = q[ii] - H[ii]

            if positive and tmp < 0:
                w[ii] = 0.0
            else:
                w[ii] = np.sign(tmp) * fmax(abs(tmp) - alpha, 0) \
                    / (Q[ii, ii] + beta)

            if w[ii] != 0.0:
                # H +=  w[ii] * Q[ii] # Update H = X.T X w
                H += w[ii] * Q[ii]
                # _axpy(n_features, w[ii], Q_ptr + ii * n_features, 1,
                #       H_ptr, 1)

            # update the maximum absolute coefficient update
            d_w_ii = abs(w[ii] - w_ii)
            if d_w_ii > d_w_max:
                d_w_max = d_w_ii

            if abs(w[ii]) > w_max:
                w_max = abs(w[ii])

        if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
            # the biggest coordinate update of this iteration was smaller than
            # the tolerance: check the duality gap as ultimate stopping
            # criterion

            # q_dot_w = np.dot(w, q)
            q_dot_w = np.dot(w, q)
            # q_dot_w = _dot(n_features, w_ptr, 1, q_ptr, 1)

            for ii in range(n_features):
                XtA[ii] = q[ii] - H[ii] - beta * w[ii]
            if positive:
                dual_norm_XtA = np.max(XtA)  # max(n_features, XtA_ptr)
            else:
                dual_norm_XtA = np.max(np.abs(XtA)) # abs_max(n_features, XtA_ptr)

            # temp = np.sum(w * H)
            tmp = 0.0
            for ii in range(n_features):
                tmp += w[ii] * H[ii]
            R_norm2 = y_norm2 + tmp - 2.0 * q_dot_w

            # w_norm2 = np.dot(w, w)
            w_norm2 = np.dot(w, w)
            # w_norm2 = _dot(n_features, &w[0], 1, &w[0], 1)

            if (dual_norm_XtA > alpha):
                const = alpha / dual_norm_XtA
                A_norm2 = R_norm2 * (const ** 2)
                gap = 0.5 * (R_norm2 + A_norm2)
            else:
                const = 1.0
                gap = R_norm2

            # The call to asum is equivalent to the L1 norm of w
            gap += (alpha * np.sum(np.abs(w)) -
                   const * y_norm2 + const * q_dot_w +
                   0.5 * beta * (1 + const ** 2) * w_norm2)
            # gap += (alpha * _asum(n_features, &w[0], 1) -
            #        const * y_norm2 +  const * q_dot_w +
            #        0.5 * beta * (1 + const ** 2) * w_norm2)

            if gap < tol:
                # return if we reached desired tolerance
                break

    else:
        # for/else, runs if for doesn't end with a `break`
        raise Warning("Objective did not converge. You might want to "
                      "increase the number of iterations. Duality "
                      "gap: {}, tolerance: {}".format(gap, tol),
                      ConvergenceWarning)


    return np.asarray(w), gap, tol, n_iter + 1
