# Project for Machine Learning with Probabilistic Programming

The Kalman Filter is the optimal time series predictor given Gaussian errors with known variances. However, in virtually all real-world scenarios, the variances are unknown. We came up with an algorithm to estimate the covariance matrices as we make predictions.

The algorithm ends up being a hybrid of sequential Monte Carlo (SMC) and variational inference (VI). Instead of taking weighted averages of particles, we make predictions using the MAP for the covariance matrices. And when the resampling step is supposed to happen, we instead update the covariance matrices.

Please read using nbviewer as the LaTeX does not render properly on GitHub.

https://nbviewer.jupyter.org/github/wicaksono-wijono/mlpp/blob/master/final-notebook.ipynb
