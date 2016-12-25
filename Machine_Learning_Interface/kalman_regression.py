import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import abc
from base_models import Regression
import matplotlib.pyplot as plt

class KalmanRegression(Regression):
    """Abstract base class for regularized regression models. 
    Do not create instances of this class for modeling!
    Use derived classes. Note that all derived classes should
    contain the attributes listed.

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    cv_folds : int
        Number of folds for cross validation. If None, no cross-validation is performed. 
    **kwargs : varies
        Keyword arguments to be passed to cross-validation function. Only appropriate if cv_folds is not None

    Attributes
    ----------
    self.intercept : boolean
        Whether to fit an intercept to the model.
    self.scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    self.cv_folds : int        
        Number of folds for cross validation
    self.coefs : pd.Series
        Fitted coefficients. Index is coefficient names. 
        Underlying fitted model. View documentation of derived classes for information
    """
    def __init__(self, intercept=False, scale=False, em=False):    
        self.intercept                                = intercept
        self.scale                                    = scale
        self.em                                       = em

    def _estimate_model(self):

        assert list(self.y_train.index) == list(self.x_train.index)

        obs_mat = (self.x_train.values.T).T[:, np.newaxis]

        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(self.number_feat)

        if self.em:
            kf = KalmanFilter(n_dim_obs=1, n_dim_state=self.number_feat,
                  transition_matrices=np.eye(self.number_feat),
                  em_vars=['transition_covariance', 'observation_covariance'],
                  observation_matrices=obs_mat)
            state_means, state_covs = kf.em(self.y_train.values).filter(self.y_train.values)
            self.params_ts = pd.DataFrame(index = self.y_train.index, data = state_means, columns = self.x_train.columns)
            self.cov_ts = state_covs
            '''
            kf = KalmanFilter(n_dim_obs=1, n_dim_state=self.number_feat,
                              transition_matrices=np.eye(self.number_feat),
                              transition_covariance=trans_cov,
                              observation_matrices=obs_mat)
            kf.em(self.y_train.values)
            self.params_ts = pd.DataFrame(index = self.y_train.index, columns = self.x_train.columns)
            self.params_ts.ix[:, :] = kf.initial_state_mean
            self.cov_ts = kf.initial_state_covariance
            '''
        else:
            kf = KalmanFilter(n_dim_obs=1, n_dim_state=self.number_feat,
                          initial_state_mean=np.zeros(self.number_feat),
                          initial_state_covariance=np.ones((self.number_feat, self.number_feat)),
                          transition_matrices=np.eye(self.number_feat),
                          transition_covariance=trans_cov,
                          observation_covariance=1.0,
                          observation_matrices=obs_mat)
            state_means, state_covs = kf.filter(self.y_train.values)
            self.betas = state_means
            self.params_ts = pd.DataFrame(index = self.y_train.index, data = state_means, columns = self.x_train.columns)
            self.cov_ts = state_covs
        return kf

    def diagnostics(self):
        super(KalmanRegression, self).diagnostics() 
        self.coefs = self._estimate_coefficients()
        self.beta_plot()

    def predict(self, x_val):
        super(KalmanRegression, self).predict(x_val) 
        self.coefs = self._estimate_coefficients()
        new_means, new_covs = self.model.filter_update(self.coefs.values,
                                                       self.cov_ts[-1],
                                                       observation_matrix = self.x_val)
        predictions = pd.Series(index=self.x_val.index, data=self.x_val.multiply(new_means).sum(axis=1), name = 'predictions')
        return predictions

    def _estimate_coefficients(self):
        coefs = pd.Series(index = self.x_train.columns, data = self.params_ts.tail(1).values[0], name = 'coefficients')
        return coefs

    def _estimate_fittedvalues(self):
        fitted_vals = pd.Series(self.x_train.multiply(self.params_ts).sum(axis=1), name = 'fitted')
        return fitted_vals

    def update(self, observation_array, y_observation):
        x_vals = observation_array
        if self.intercept:
            x_vals = np.insert(x_vals, 0, 1.0)

        obs_mat = np.expand_dims(x_vals, 0)
        obs = np.expand_dims(y_observation, 0)

        new_means, new_covs = self.model.filter_update(self.params_ts.values,
                                                       self.cov_ts[-1],
                                                       observation_matrix = obs_mat,
                                                       observation = obs)

        self.params = pd.Series(index = self.x_df.columns, data = new_means.data, name = 'params')
        self.cov_ts = np.append(self.cov_ts, new_covs)       

    def beta_plot(self):
        plt.figure()
        plt.plot(self.params_ts)
        plt.xlabel('Date')
        plt.ylabel('Coefficient')
        plt.title('Coefficients Over Time')
        return plt