import numpy as np
import scipy.linalg
import scipy.sparse
import warnings

def varpro2_expfun(alpha, t):
    A = t[:,np.newaxis] @ alpha[:,np.newaxis].T
    return np.exp(A)


def varpro2_dexpfun(alpha, t, j):
    # computes d/d(alpha_i) where we begin indexing at 0
    if (j < 0) or (j >= len(alpha)):
        raise ValueError("varpro2_dexpfun: cannot compute %sth derivative. Index j for d/d(alpha_j) out of range."%j)
    t = t.reshape((-1, 1))
    A = scipy.sparse.lil_matrix((t.size, alpha.size), dtype=complex)
    A[:, j] = t * np.exp(alpha[j] * t)
    return scipy.sparse.csc_matrix(A)


def varpro2_opts(set_options_dict=None):
    options = {
        "lambda0": 1.0,
        "max_lambda": 52,
        "lambda_up": 2.0,
        "lambda_down": 3.0,
        "use_marquardt_scaling": True,
        "max_iterations": 30,
        "tolerance": 1.0e-1,
        "eps_stall": 1.0e-12,
        "compute_full_jacobian": True,
        "verbose": True,
        "ptf": 1
    }
    optionsmin = {
        "lambda0": 0.0,
        "max_lambda": 0,
        "lambda_up": 1.0,
        "lambda_down": 1.0,
        "use_marquardt_scaling": False,
        "max_iterations": 0,
        "tolerance": 0.0,
        "eps_stall": -np.finfo(np.float64).min,
        "compute_full_jacobian": False,
        "verbose": False,
        "ptf": 0
    }
    optionsmax = {
        "lambda0": 1.0e16,
        "max_lambda": 200,
        "lambda_up": 1.0e16,
        "lambda_down": 1.0e16,
        "use_marquardt_scaling": True,
        "max_iterations": 1.0e12,
        "tolerance": 1.0e16,
        "eps_stall": 1.0,
        "compute_full_jacobian": True,
        "verbose": True,
        "ptf": 2147483647                 # sys.maxsize() for int datatype
    }
    if not set_options_dict:
        print("Default varpro2 options used.")
    else:
        for key in set_options_dict:
            if key in options:
                if optionsmin[key] <= set_options_dict[key] <= optionsmax[key]:
                    options[key] = set_options_dict[key]
                else:
                    warnings.warn("Value %s = %s is not in valid range (%s,%s)" %
                                  (key, set_options_dict[key], optionsmin[key], optionsmax[key]), Warning)
            else:
                warnings.warn("Key %s not in options" % key, Warning)
    return options


def varpro2(y, t, phi_function, dphi_function, alpha_init,
            linear_constraint=False,
            tikhonov_regularization=0,
            prox_operator=False,
            options=None):
    """
    :param y: data matrix
    :param t: vector of sample times
    :param phi: function phi(alpha,t) that takes matrices of size (m,n)
    :param dphi: function dphi(alpha,t,i) returning the d(phi)/d(alpha)
    :param alpha_init: initial guess for vector alpha
    :param use_tikhonov_regularization: Sets L2 regularization. Zero or False will have no L2 regularization.
    Can use a scalar (gamma) or matrix: min|y - phi*b|_F^2 + |gamma alpha|_2^2
    :param prox_operator: prox operator that can be applied to vector alpha at each step
    :param options: options for varpro2
    """

    def update_alpha(alpha, rjac, rhs, djacobian_pivot, prox_operator):
        # update eigenvalues
        delta0 = scipy.linalg.lstsq(rjac, rhs)[0]
        delta0 = delta0[djacobian_pivot]
        alpha0 = alpha + delta0
        if prox_operator:
            alpha0 = prox_operator(alpha0)
            delta0 = alpha0 - alpha
        return alpha0, delta0

    def varpro2_solve(phi, y, gamma, alpha):
        # least squares solution for mode amplitudes phi @ b = y, residual, and error
        b = scipy.linalg.lstsq(phi, y)[0]
        residual = y - phi@b
        if len(alpha) == 1 or np.isscalar(alpha):
            alpha = np.ravel(alpha).item()*np.eye(*gamma.shape)
        error_last = 0.5*(np.linalg.norm(residual, 'fro')**2 + np.linalg.norm(gamma@alpha)**2)
        return b, residual, error_last

    def varpro2_svd(phi, tolrank):
        # rank truncated svd where rank is scaled by a tolerance
        U, s, Vh = np.linalg.svd(phi, full_matrices=False)
        rank = np.sum(s > tolrank*s[0])
        U = U[:, :rank]
        s = s[:rank]
        V = Vh[:rank, :].conj().T
        return U, s, V

    t = np.ravel(t)
    n_data_cols = y.shape[1]
    n_t = len(t)
    n_alpha = len(alpha_init)

    options = varpro2_opts(set_options_dict=options)
    lambda0 = options['lambda0']

    if linear_constraint:
        # TODO linear constraints functionality
        raise Exception("linear constraint functionality not yet coded!")

    if tikhonov_regularization:
        if np.isscalar(tikhonov_regularization):
            gamma = tikhonov_regularization*np.eye(n_alpha)
    else:
        gamma = np.zeros((n_alpha, n_alpha))

    if prox_operator:
        alpha_init = prox_operator(alpha_init)

    # Initialize values
    alpha = np.copy(np.asarray(alpha_init, dtype=complex))
    alphas = np.zeros((n_alpha, options['max_iterations']), dtype=complex)
    if tikhonov_regularization:
        djacobian = np.zeros((n_t*n_data_cols + n_alpha, n_alpha), dtype=complex)
        rhs_temp = np.zeros(n_t*n_data_cols + n_alpha, dtype=complex)
        raise Exception("Tikhonov part not coded")
    else:
        djacobian = np.zeros((n_t*n_data_cols, n_alpha), dtype=complex)
        rhs_temp = np.zeros(n_t*n_data_cols, dtype=complex)
    error = np.zeros(options['max_iterations'])
    # res_scale = np.linalg.norm(y, 'fro')      # TODO res_scale unused in Askham's MATLAB code. Ditch it?
    scales = np.zeros(n_alpha)
    rjac = np.zeros((2*n_alpha, n_alpha), dtype=complex)

    phi = phi_function(alpha, t)
    tolrank = n_t*np.finfo(float).eps
    U, s, V = varpro2_svd(phi, tolrank)
    b, residual, error_last = varpro2_solve(phi, y, gamma, alpha)

    for iteration in range(options['max_iterations']):
        # build jacobian matrix by looping over alpha indices
        for j in range(n_alpha):
            dphi_temp = dphi_function(alpha, t, j)  # d/(dalpha_j) of phi. sparse output.
            sp_U = scipy.sparse.csc_matrix(U)
            djacobian_a = (dphi_temp - sp_U @ (sp_U.conj().T @ dphi_temp)).todense() @ b
            if options['compute_full_jacobian']:
                djacobian_b = U@scipy.linalg.lstsq(np.diag(s), V.conj().T @ dphi_temp.conj().T.todense() @ residual)[0]
                djacobian[:n_t*n_data_cols, j] = djacobian_a.ravel(order='F') + djacobian_b.ravel(order='F')
            else:
                djacobian[:n_t*n_data_cols, j] = djacobian_a.A.ravel(order='F')  # approximate Jacobian
            if options['use_marquardt_scaling']:
                scales[j] = min(np.linalg.norm(djacobian[:n_t*n_data_cols, j]), 1.0)
                scales[j] = max(scales[j], 1e-6)
            else:
                scales[j] = 1.0

        if tikhonov_regularization:
            print("using tikhonov regularization")
            djacobian[n_t*n_data_cols + 1:, :] = gamma

        # loop to determine lambda for the levenberg part
        # precompute components that don't depend on step-size parameter lambda
        # get pivots and lapack style qr for jacobian matrix
        rhs_temp[:n_t*n_data_cols] = residual.ravel(order='F')

        if tikhonov_regularization:
            rhs_temp[n_t*n_data_cols:] = -gamma@alpha

        g = djacobian.conj().T@rhs_temp

        djacobian_Q, djacobian_R, djacobian_pivot = scipy.linalg.qr(djacobian, mode='economic',
                                                                    pivoting=True)  # TODO do i need householder reflections?
        rjac[:n_alpha, :] = np.triu(djacobian_R[:n_alpha, :])
        rhs_top = djacobian_Q.conj().T@rhs_temp
        rhs = np.concatenate((rhs_top[:n_alpha], np.zeros(n_alpha)), axis=0)

        scales_pivot = scales[djacobian_pivot]
        rjac[n_alpha:2*n_alpha, :] = lambda0*np.diag(scales_pivot)

        alpha0, delta0 = update_alpha(alpha, rjac, rhs, djacobian_pivot, prox_operator)
        phi = phi_function(alpha0, t)
        b0, residual0, error0 = varpro2_solve(phi, y, gamma, alpha0)

        # update rule
        actual_improvement = error_last - error0
        predicted_improvement = np.real(0.5*delta0.conj().T@g)
        improvement_ratio = actual_improvement/predicted_improvement

        descent = " "  # marker that indicates in output whether the algorithm needed to enter the descent loop
        if error0 < error_last:
            # rescale lambda based on actual vs pred improvement
            lambda0 = lambda0*max(1/options['lambda_down'], 1 - (2*improvement_ratio - 1)**3)
            alpha, error_last, b, residual = (alpha0, error0, b0, residual0)
        else:
            # increase lambda until something works. kinda like gradient descent
            descent = "*"
            for j in range(options['max_lambda']):
                lambda0 = lambda0*options['lambda_up']
                rjac[n_alpha:2*n_alpha, :] = lambda0*np.diag(scales_pivot)

                alpha0, delta0 = update_alpha(alpha, rjac, rhs, djacobian_pivot, prox_operator)
                phi = phi_function(alpha0, t)
                b0, residual0, error0 = varpro2_solve(phi, y, gamma, alpha0)
                if error0 < error_last:
                    alpha, error_last, b, residual = (alpha0, error0, b0, residual0)
                    break

            if error0 > error_last:
                error[iteration] = error_last
                convergence_message = "Failed to find appropriate step length at iteration %d. Residual %s. Lambda %s"%(
                iteration, error_last, lambda0)
                if options['verbose']:
                    warnings.warn(convergence_message, Warning)
                return b, alpha, alphas, error, iteration, (False, convergence_message)

        # update and status print
        alphas[:, iteration] = alpha
        error[iteration] = error_last
        if options['verbose'] and (iteration%options['ptf'] == 0):
            print("step %02d%s error %.5e lambda %.5e"%(iteration, descent, error_last, lambda0))

        if error_last < options['tolerance']:
            convergence_message = "Tolerance %s met"%options['tolerance']
            return b, alpha, alphas, error, iteration, (True, convergence_message)

        if iteration > 0:
            if error[iteration - 1] - error[iteration] < options['eps_stall']*error[iteration - 1]:
                convergence_message = "Stall detected. Residual reduced by less than %s times previous residual."%(
                options['eps_stall'])
                if options['verbose']:
                    print(convergence_message)
                return b, alpha, alphas, error, iteration, (True, convergence_message)
            pass

        phi = phi_function(alpha, t)
        U, s, V = varpro2_svd(phi, tolrank)

    convergence_message = "Failed to reach tolerance %s after maximal %d iterations. Residual %s"%(
    options['tolerance'], iteration, error_last)
    if options['verbose']:
        warnings.warn(convergence_message, Warning)
    return b, alpha, alphas, error, iteration, (False, convergence_message)