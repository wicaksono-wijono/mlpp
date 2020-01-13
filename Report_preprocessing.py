import numpy as np
import pandas as pd

path = './dataset/toyset/'
filename = 'processed_toyset.csv'
df = pd.read_csv(path + filename)

# #### Define dataset row/column sizes to begin with

df.loc[:, 'gbpusd'] = df.loc[:, 'gbpusd']*100
df.loc[:, 'gbpjpy'] = df.loc[:, 'gbpjpy']*100
df.loc[:, 'usdjpy'] = df.loc[:, 'usdjpy']*100

nrow = 3
ncol = 3


# Some scratch codes

proc_covar = 0.0001**2*np.identity(nrow)
post_covar = proc_covar
meas_covar = 0.0003**2*np.identity(nrow)
pred_covar = meas_covar
ident = np.identity(nrow)


state_old = np.ones((3, 1), dtype=np.float64)
emit_mat = np.array([[1., -1,  0.],
                     [1.,  0, -1.],
                     [0., -1., 1.]],
                    dtype=np.float64
                    )
obs_mat = df.drop(['timestamp', 'time_gap'], axis=1)
obs_mat = obs_mat.to_numpy()
time_gap = df.time_gap.to_numpy().reshape(-1, 1)

latent_states = np.empty(shape=(df.shape[0], nrow), dtype=np.float64)
predicted_obs = np.empty(shape=(df.shape[0], nrow), dtype=np.float64)


ccy_list = ['gbpjpy', 'gbpusd', 'usdjpy']
latent_cols = ['GBP', 'JPY', 'USD']


for i in range(df.shape[0]):
    post_covar = post_covar + time_gap[i]*proc_covar

    # obs_mat[i,:] is a (3,) vector. So, we need flatten() to match
    innovation = obs_mat[i, :] - (emit_mat@state_old).flatten()
    innovation_covar = emit_mat@post_covar@(emit_mat.T) + meas_covar
    kalman_gain = post_covar@(emit_mat.T)@np.linalg.inv(innovation_covar)

    # We transpose (kalman_gain@innovation) from (3,) to (3,1)
    state_new = state_old + (kalman_gain@innovation).reshape(-1, 1)
    post_covar = (ident - kalman_gain@emit_mat)@post_covar
    predicted_obs[i] = (emit_mat@state_new).T
    latent_states[i] - state_new.T
    state_old = state_new


new_idx = df.timestamp.shift(-1).to_numpy().reshape(-1, 1)
predictions = np.hstack((new_idx, predicted_obs))
latent_estimates = np.hstack((new_idx, latent_states))


assert predictions.shape[0] == predicted_obs.shape[0], "We have data loss"
assert latent_estimates.shape[0] == latent_states.shape[0], "We have data loss"


# #### Convert arrays to DataFrame instances
predictions = pd.DataFrame(predictions, columns=['timestamp'] + ccy_list)
latent_estimates = pd.DataFrame(latent_estimates,
                                columns=['timestampe'] + latent_cols)
