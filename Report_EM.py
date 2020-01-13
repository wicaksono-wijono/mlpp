import numpy as np
import pandas as pd

# ## Parameter Estimation: EM

# So the goal of the project is to build an adaptive Kalman Filter. However,
# the problem is impossible at our current level, so we have to do a middle
# ground. A Kalman Smoother allows us to estimate the covariance matrices
# through the EM algorithm, but it requires using future data to predict
# the past. By combining a Kalman Smoother with PSIS detailed in the next
# section, we construct an algorithm to do filtering in a heuristic fashion.

# ### E step
# - The forward pass propagates through the system, given fixed
# covariance matrices.
# - It outputs both the predicted states and the predicted measurements.

nrow = 3
ncol = 3
ident = np.identity(nrow)

ccy_list = ['gbpjpy', 'gbpusd', 'usdjpy']
latent_cols = ['GBP', 'JPY', 'USD']


def forward_pass(data, initial_state, emit_mat, meas_covar,
                 proc_covar, post_covar):

    '''
    Parameters:
      data(DataFrame, ): a (batch) dataset
    '''

    if type(data) == pd.core.series.Series:
        obs_mat = data[1:-1]
        obs_mat = obs_mat.to_numpy().reshape(1, -1)
        time_gap = np.array(data[-1]).reshape(-1, 1)
        data_nrow = 1
        new_idx = np.datetime64(data.timestamp) + np.timedelta64(data[-1], 'm')
        new_idx = np.array(str(new_idx)).reshape(-1, 1)
    else:
        obs_mat = data.drop(['timestamp', 'time_gap'], axis=1)
        obs_mat = obs_mat.to_numpy()
        time_gap = data.time_gap.to_numpy().reshape(-1, 1)
        data_nrow = data.shape[0]
        new_idx = data.timestamp.shift(-1).to_numpy().reshape(-1, 1)

    latent_states = np.empty((data_nrow, nrow))
    predicted_obs = np.empty((data_nrow, nrow))

    state_old = initial_state[1:-1] \
        if initial_state.shape[0] > nrow else initial_state

    for i in range(data_nrow):
        # post_covar = post_covar + time_gap.flatten()[i]*proc_covar
        post_covar = post_covar + time_gap.flatten()[i]*proc_covar

        # obs_mat[i,:] is a (3,) vector. So, we need flatten() to match
        innovation = obs_mat[i, :] - (emit_mat@state_old).flatten()
        innovation_covar = emit_mat@post_covar@(emit_mat.T) + meas_covar

        kalman_gain = post_covar@(emit_mat.T)@(
            np.linalg.inv(innovation_covar).astype(np.float64))

        # We transpose (kalman_gain@innovation) from (3,) to (3,1)
        state_new = state_old.reshape(-1, 1) + \
            (kalman_gain@innovation).reshape(-1, 1)
        post_covar = (ident - kalman_gain@emit_mat)@post_covar
        predicted_obs[i] = (emit_mat@state_new).T
        latent_states[i] = state_new.T
        state_old = state_new

    predicted_obs = np.array(predicted_obs)
    latent_states = np.array(latent_states)
    predictions = np.hstack((new_idx, predicted_obs))
    predictions = np.hstack((predictions, time_gap))
    latent_estimates = np.hstack((new_idx, latent_states))
    latent_estimates = np.hstack((latent_estimates, time_gap))

    return predictions, latent_estimates, post_covar, innovation_covar


# A helper function.
def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e


# ### M step
def update_proc_scale(latent_estimates):
    '''
    Takes the scale matrix of the propagated states.
    But we have to take into account the uneven time gaps.
    '''

    latent_estimates = latent_estimates[:-1, :]  # drop nan
    time_gap_rep = np.repeat(latent_estimates[:, -1].astype(np.float64),
                             repeats=nrow).reshape(-1, nrow)
    latent_estimates = latent_estimates[:, 1:-1]
    latent_estimates = (latent_estimates - shift(latent_estimates, 1)) \
        .astype(np.float64) / time_gap_rep
    latent_estimates = \
        latent_estimates[~np.isnan(latent_estimates)].reshape(-1, nrow)

    proc_scale = (latent_estimates.T)@latent_estimates

    return proc_scale


def update_meas_scale(pred, obs):
    '''
    Gets us the observed scale matrix through (biased) MLE.
    The debiasing is done by supplying the correct df to the invWishart
    '''
    pred = pred[~pd.isna(pred[:, 0]), :]
    obs_timestamp = obs['timestamp'].dropna(how='any').to_numpy()
    unique_timestamp = np.union1d(pred[:, 0], obs_timestamp)

    pred_mat = pred[np.isin(pred[:, 0], unique_timestamp)]
    obs_mat = obs[np.isin(obs_timestamp, unique_timestamp)]

    # Convert np to dataframe
    pred_mat = pd.DataFrame(pred_mat, columns=['timestamp'] +
                            ccy_list + ['time_gap'])
    obs_pred_mat = pd.merge(left=obs_mat, right=pred_mat,
                            how='inner', on='timestamp',
                            suffixes=('_obs', '_pred'))

    ccy_list_obs = [ccy+'_obs' for ccy in ccy_list]
    ccy_list_pred = [ccy+'_pred' for ccy in ccy_list]

    error_mat = np.array(obs_pred_mat[ccy_list_pred], dtype=np.float64) - \
        np.array(obs_pred_mat[ccy_list_obs], dtype=np.float64)

    return (error_mat.T)@error_mat


# ### EM (Kalman smoother)


def kalman_smoother(data, prior_state, emit_mat, post_covar,
                    prior_meas_df, prior_meas_scale, prior_proc_df,
                    prior_proc_scale):
    '''
    Runs the smoother on a time block through EM
    '''
    nrow_data = data.shape[0]
    predicted_measurements, states,\
        ps_covar, innovation_covar = \
        forward_pass(data=data, initial_state=prior_state,
                     emit_mat=emit_mat,
                     meas_covar=prior_meas_scale/prior_meas_df,
                     proc_covar=prior_proc_scale/prior_proc_df,
                     post_covar=post_covar)

    post_proc_scale = update_proc_scale(states)
    post_meas_scale = update_meas_scale(pred=predicted_measurements, obs=data)

    old_proc_scale = post_proc_scale*2
    old_meas_scale = post_meas_scale*2

    while np.sqrt(np.sum(((old_proc_scale - post_proc_scale)/nrow_data)**2) +
                  np.sum(((old_meas_scale-post_meas_scale)/nrow_data)**2)) > \
            10**(-9):

        # print(np.sqrt(np.sum(((old_proc_scale - post_proc_scale) /
        #       nrow_data)**2) +
        #       np.sum(((old_meas_scale-post_meas_scale)/nrow_data)**2)))

        old_meas_scale = post_meas_scale
        old_proc_scale = post_proc_scale

        predicted_measurements, states, ps_covar, innovation_covar = \
            forward_pass(data=data, initial_state=prior_state,
                         emit_mat=emit_mat,
                         meas_covar=post_meas_scale/nrow_data,
                         proc_covar=post_proc_scale/nrow_data,
                         post_covar=post_covar)

        post_proc_scale = update_proc_scale(states)
        post_meas_scale = update_meas_scale(
            pred=predicted_measurements, obs=data)

    post_proc_df = prior_proc_df + nrow_data
    post_meas_df = prior_meas_df + nrow_data

    result = {
        'input_data': data,
        'states': states,
        'predicted_measurements': predicted_measurements,
        'post_meas_df': post_meas_df,
        'post_meas_scale': prior_meas_scale + post_meas_scale,
        'post_proc_df': post_proc_df,
        'post_proc_scale': prior_proc_scale + post_proc_scale,
        'post_covar': ps_covar,
        'innovation_covar': innovation_covar
    }

    return result
