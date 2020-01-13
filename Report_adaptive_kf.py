from scipy.stats import invwishart, multivariate_normal
import numpy as np

from Report_EM import forward_pass, kalman_smoother, shift

nrow = 3
ncol = 3


def kalman_filter_minibatch(data, emit_mat, initial_state,
                            innovation_covar, post_covar,
                            prior_proc_df, prior_proc_scale,
                            prior_meas_df, prior_meas_scale
                            ):
    batch_size = 200
    data_nrow = data.shape[0]
    proc_cov_mats = invwishart.rvs(
        df=prior_proc_df, scale=prior_proc_scale, size=batch_size)
    meas_cov_mats = invwishart.rvs(
        df=prior_meas_df, scale=prior_meas_scale, size=batch_size)

    psis_k = 0
    iter_count = 0
    initial_state = initial_state[1:-1] \
        if initial_state.shape[0] > nrow else initial_state
    parallel_states = np.expand_dims(initial_state, axis=0)
    parallel_states = parallel_states.repeat(batch_size, axis=0)
    innovation_covar_mats = np.expand_dims(innovation_covar, axis=0)
    innovation_covar_mats = innovation_covar_mats.repeat(batch_size, axis=0)
    post_cov_mats = np.expand_dims(post_covar, axis=0)
    post_cov_mats = post_cov_mats.repeat(batch_size, axis=0)

    innovation_covar = []
    importance_weights = []

    while (psis_k < 0.7) and (iter_count < data_nrow):
        for i in range(parallel_states.shape[0]):
            temp1, temp2, temp3, temp4 = \
                forward_pass(data=data.iloc[iter_count, :],
                             initial_state=initial_state,
                             emit_mat=emit_mat,
                             meas_covar=meas_cov_mats[i, :, :],
                             proc_covar=proc_cov_mats[i, :, :],
                             post_covar=post_cov_mats[i, :, :])
            parallel_states[i] = temp2[-1, 1:-1]
            innovation_covar_mats[i] = temp4
            post_cov_mats[i] = temp3

        iw = []
        for i in range(parallel_states.shape[0]):
            iw.append(
                multivariate_normal.logpdf(
                    x=(data.iloc[iter_count, 1:-1]).astype(np.float64),
                    mean=emit_mat@(parallel_states[i, :].astype(np.float64)),
                    cov=innovation_covar_mats[i, :, :]))
        importance_weights.append(iw)

        psis_weights = np.array(importance_weights)
        psis_weights = np.sum(psis_weights, axis=0)

        psis_weights = psis_weights[np.argsort(
            psis_weights)[-int(batch_size*0.2):]]
        psis_weights = np.exp(psis_weights)

        psis_k = gpdfitnew(psis_weights - np.min(psis_weights))[0]
        # print('iter_count', iter_count, 'k:', psis_k)
        iter_count += 1

    forward = forward_pass(
        data=data[:iter_count],
        initial_state=initial_state,
        emit_mat=emit_mat,
        meas_covar=prior_meas_scale / (prior_meas_df - emit_mat.shape[0] - 1),
        proc_covar=prior_proc_scale / (prior_proc_df - emit_mat.shape[1] - 1),
        post_covar=post_covar
    )

    return forward


def adaptive_kalman_filter(data, emit_mat, prior_state, prior_covar,
                           innovation_covar,
                           prior_proc_df, prior_proc_scale, prior_meas_df,
                           prior_meas_scale, burn_in=0):
    # print('Adaptive KF started with a prior state of', prior_state)
    output_states = []
    output_meas = []
    output_meas_covar = []
    output_proc_covar = []
    breakpoints = []
    minibatch_count = 0
    itercount = 0
    nrow_data = data.shape[0]

    while itercount < nrow_data:
        forward1, forward2, forward3, forward4 = kalman_filter_minibatch(
            data=data[itercount:nrow_data],
            emit_mat=emit_mat,
            initial_state=prior_state,
            innovation_covar=innovation_covar,
            post_covar=prior_covar,
            prior_proc_df=prior_proc_df,
            prior_proc_scale=prior_proc_scale,
            prior_meas_df=prior_meas_df,
            prior_meas_scale=prior_meas_scale
        )

        batch_size = forward1.shape[0]
        # print('batch_size', batch_size)
        if batch_size + itercount >= nrow_data:
            break

        smoother = kalman_smoother(
            data=data[itercount:min(itercount+batch_size, nrow_data)],
            prior_state=prior_state,
            emit_mat=emit_mat,
            post_covar=prior_covar,
            prior_meas_df=prior_meas_df,
            prior_meas_scale=prior_meas_scale,
            prior_proc_df=prior_proc_df,
            prior_proc_scale=prior_proc_scale
        )

        minibatch_count = minibatch_count + 1
        itercount = itercount + batch_size

        breakpoints.append(itercount)

        output_states.append(forward2)
        output_meas.append(forward1)
        output_meas_covar.append(
            smoother['post_meas_scale'] / (smoother['post_meas_df']
                                           - emit_mat.shape[0] - 1))
        output_proc_covar.append(
            smoother['post_proc_scale'] / (smoother['post_proc_df']
                                           - emit_mat.shape[1] - 1))

        batch_size = smoother['input_data'].shape[0]

        prior_state = forward2
        prior_state = prior_state[-1, :].astype(np.float64)

        prior_meas_df = smoother['post_meas_df']
        prior_meas_scale = smoother['post_meas_scale']

        prior_proc_df = smoother['post_proc_df']
        prior_proc_scale = smoother['post_proc_scale']

        prior_covar = forward3
        innovation_covar = forward4

        # print('itercount', itercount)
        # end of while

    # print("A loop has finished in adaptive_kalman_filter.")

    output_states_combined = np.vstack(output_states)
    n_states = output_states_combined.shape[0]
    output_states_combined[:, 0] = shift(data.iloc[:n_states, 0], -1)

    output_meas_combined = np.vstack(output_meas)
    n_meas = output_meas_combined.shape[0]
    output_meas_combined[:, 0] = shift(data.iloc[:n_meas, 0], -1)

    result = {
        'input_data': data,
        'states': output_states_combined,
        'predicted_meas': output_meas_combined,
        'post_meas_df': prior_meas_df,
        'post_meas_scale': prior_meas_scale,
        'post_proc_df': prior_proc_df,
        'post_proc_scale': prior_proc_scale,
        'post_covar': prior_covar,
        'innovation_covar': innovation_covar,
        'breakpoints': breakpoints,
        'output_meas_covar': output_meas_covar,
        'output_proc_covar': output_proc_covar
    }

    return result


# Source: https://github.com/avehtari/PSIS/blob/master/py/psis.py
def gpdfitnew(x, sort=True, sort_in_place=False, return_quadrature=False):
    """Estimate the paramaters for the Generalized Pareto Distribution (GPD)
    Returns empirical Bayes estimate for the parameters of the two-parameter
    generalized Parato distribution given the data.
    Parameters
    ----------
    x : ndarray
        One dimensional data array
    sort : bool or ndarray, optional
        If known in advance, one can provide an array of indices that would
        sort the input array `x`. If the input array is already sorted, provide
        False. If True (default behaviour), the array is sorted internally.
    sort_in_place : bool, optional
        If `sort` is True and `sort_in_place` is True, the array is sorted
        in-place (False by default).
    return_quadrature : bool, optional
        If True, quadrature points and weight `ks` and `w` of the marginal
        posterior distribution of k are also calculated and returned. False by
        default.
    Returns
    -------
    k, sigma : float
        estimated parameter values
    ks, w : ndarray
        Quadrature points and weights of the marginal posterior distribution
        of `k`. Returned only if `return_quadrature` is True.
    Notes
    -----
    This function returns a negative of Zhang and Stephens's k, because it is
    more common parameterisation.
    """
    if x.ndim != 1 or len(x) <= 1:
        raise ValueError("Invalid input array.")

    # check if x should be sorted
    if sort is True:
        if sort_in_place:
            x.sort()
            xsorted = True
        else:
            sort = np.argsort(x)
            xsorted = False
    elif sort is False:
        xsorted = True
    else:
        xsorted = False

    n = len(x)
    PRIOR = 3
    m = 30 + int(np.sqrt(n))

    bs = np.arange(1, m + 1, dtype=float)
    bs -= 0.5
    np.divide(m, bs, out=bs)
    np.sqrt(bs, out=bs)
    np.subtract(1, bs, out=bs)
    if xsorted:
        bs /= PRIOR * x[int(n/4 + 0.5) - 1]
        bs += 1 / x[-1]
    else:
        bs /= PRIOR * x[sort[int(n/4 + 0.5) - 1]]
        bs += 1 / x[sort[-1]]

    ks = np.negative(bs)
    temp = ks[:, None] * x
    np.log1p(temp, out=temp)
    np.mean(temp, axis=1, out=ks)

    L = bs / ks
    np.negative(L, out=L)
    np.log(L, out=L)
    L -= ks
    L -= 1
    L *= n

    temp = L - L[:, None]
    np.exp(temp, out=temp)
    w = np.sum(temp, axis=1)
    np.divide(1, w, out=w)

    # remove negligible weights
    dii = w >= 10 * np.finfo(float).eps
    if not np.all(dii):
        w = w[dii]
        bs = bs[dii]
    # normalise w
    w /= w.sum()

    # posterior mean for b
    b = np.sum(bs * w)
    # Estimate for k, note that we return a negative of Zhang and
    # Stephens's k, because it is more common parameterisation.
    temp = (-b) * x
    np.log1p(temp, out=temp)
    k = np.mean(temp)
    if return_quadrature:
        np.negative(x, out=temp)
        temp = bs[:, None] * temp
        np.log1p(temp, out=temp)
        ks = np.mean(temp, axis=1)
    # estimate for sigma
    sigma = -k / b * n / (n - 0)
    # weakly informative prior for k
    a = 10
    k = k * n / (n+a) + a * 0.5 / (n+a)
    if return_quadrature:
        ks *= n / (n+a)
        ks += a * 0.5 / (n+a)

    if return_quadrature:
        return k, sigma, ks, w
    else:
        return k, sigma
