import numpy as np
np.random.seed(10)
from timeit import default_timer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class TimeTesting(object):
    def __init__(self, model):
        self.model      = model

    ####Generates Simple Linear Regression Data
    def gen_data(self, n,p, sigma, sparsity):
        mean_array = np.random.randn(p)
        x_data = np.zeros([n, p])
        for var in range(mean_array.shape[0]):
            x_data[:,var] = np.random.normal(mean_array[var], 1, n)
        error = np.random.normal(0, sigma, n)
        beta = np.random.randn(p)
        zero_vars = np.random.choice(p,int(p*sparsity),replace=False)
        beta[zero_vars] = 0
        y_data = np.dot(x_data, beta) + error
        return x_data, y_data, beta

    ######Testing Big Data (p and n both increasing, p/n ratio constant at .1)
    def model_times(self, n_samples, p_samples, sparsity, sigma=.1):
        times = []
        for n, p, sparse in zip(n_samples, p_samples, sparsity):
            x_data, y_data, beta = self.gen_data(int(n), int(p), sigma, sparse)
            start = default_timer()
            self.model.fit(pd.DataFrame(x_data), pd.DataFrame(y_data))
            duration = default_timer() - start
            times.append(duration)
        return times

    def standard_tests(self):
        ###Testing Scale: n increasing
        times_scale = self.model_times(n_samples=np.logspace(2, 7, 6), p_samples=[10]*6, sparsity=[0]*6)

        ###Testing Dimensionality: p increasing
        times_dim = self.model_times(n_samples=[100]*5, p_samples=np.logspace(2, 6, 5), sparsity=[0]*5)

        ###Testing High Dimensional Data: n and p increasing, n/p ratio constant .1
        times_hd = self.model_times(n_samples=np.logspace(2, 4, 3), p_samples=np.logspace(1, 3, 3), sparsity=[0]*3)  ###Add more to this

        ###Testing Sparsity
        times_sparse = self.model_times(n_samples=[10000]*10, p_samples=[100]*10, sparsity=np.linspace(0,.9,10))  ###Is this necessary? Doesn't really vary

        ###Collecting Results
        self.times_dict = {'scale' : times_scale, 'dim' : times_dim, 'hd' : times_hd, 'sparse' : times_sparse}
        self.params_dict = {'scale' : np.logspace(2, 7, 6), 'dim' : np.logspace(2, 6, 5), 'hd' : np.logspace(2, 4, 3), 'sparse' : np.linspace(0,.9,10)}
        return self.times_dict

    def plot_times(self): ###Fix the labels on the graphs
        for time_key in self.times_dict:
            plt.figure()
            plt.title('Model Fitting Time as a Function of Data Size: ' + time_key)
            plt.ylabel('Time to Fit Model in Seconds')
            plt.xlabel('Number of Data Points')
            plt.plot(self.params_dict[time_key], self.times_dict[time_key])
            plt.show()

    ###Final function- calculate and store values for all major regression approaches, use as benchmarks for current

class AccuracyTesting(object):
    def __init__(self, model):
        self.model      = model

    ####Generates Simple Linear Regression Data
    def gen_data(self, n,p, sigma, sparsity):
        mean_array = np.random.randn(p)
        x_data = np.zeros([n, p])
        for var in range(mean_array.shape[0]):
            x_data[:,var] = np.random.normal(mean_array[var], 1, n)
        error = np.random.normal(0, sigma, n)
        beta = np.random.randn(p)
        zero_vars = np.random.choice(p,int(p*sparsity),replace=False)
        beta[zero_vars] = 0
        y_data = np.dot(x_data, beta) + error
        return x_data, y_data, beta

    ####Generates Simple Linear Regression Data for varying levels of correlation between all variables (much slower than non-correlation)
    def gen_data_corr(self, n, p, sigma, sparsity, cov, pair=False):
        if pair: ###Only make correlation for 2 variables
            cov_mat = np.diag([1]*p)
            cov_mat[0,1] = cov
            cov_mat[1,0] = cov   
        else:
            cov_mat = np.zeros([p,p])
            cov_mat.fill(cov)
            np.fill_diagonal(cov_mat,1)
        mean   = np.random.randn(p)
        x_data = np.random.multivariate_normal(mean, cov_mat, n)
        error = np.random.normal(0, sigma, n)
        beta = np.random.multivariate_normal([0]*p, cov_mat)
        zero_vars = np.random.choice(p,int(p*sparsity),replace=False)
        beta[zero_vars] = 0
        y_data = np.dot(x_data, beta) + error
        return x_data, y_data, beta

    ######Testing Big Data (p and n both increasing, p/n ratio constant at .1)
    def model_accuracy(self, n_samples, p_samples, sparsity, sigma, cov_array, pair=False, corr=False):
        times = []
        mse = []
        beta_err = []
        for n, p, sparse, sig, cov in zip(n_samples, p_samples, sparsity, sigma, cov_array):
            if corr:
                x_data, y_data, beta = self.gen_data_corr(int(n), int(p), sig, sparse, cov, pair)
            else:
                x_data, y_data, beta = self.gen_data(int(n), int(p), sig, sparse)
            start = default_timer()
            self.model.fit(pd.DataFrame(x_data), pd.Series(y_data))
            duration = default_timer() - start
            times.append(duration)
            y_pred = model.predict(pd.DataFrame(x_data))
            err = mean_squared_error(y_data, y_pred)
            mse.append(err)
            model.diagnostics()
            beta_accuracy = mean_squared_error(model.coefs[:-1],beta)
            beta_err.append(beta_accuracy)
        return times, mse, beta_err

    def standard_tests(self):
        ###Testing Scale: n increasing
        self.times_scale, self.mse_scale, self.beta_err_scale = self.model_accuracy(n_samples=np.logspace(2, 6, 5), p_samples=[10]*5, sparsity=[0]*5, sigma=[.1]*5, cov_array=[0]*5)

        ###Testing Sparsity
        self.times_sparse, self.mse_sparse, self.beta_err_sparse = self.model_accuracy(n_samples=[10000]*10, p_samples=[100]*10, sparsity=np.linspace(0,.9,10), sigma=[.1]*10, cov_array=[0]*10) 

        ###Testing Multicollinearity
        self.times_scale_corr, self.mse_scale_corr, self.beta_err_corr = self.model_accuracy(n_samples=[10000]*10, p_samples=[100]*10, sparsity=[0]*10, sigma=[.1]*10, cov_array=np.linspace(0, .9, 10), corr=True) 

        ###Testing Noise Level
        self.times_scale_noise, self.mse_scale_noise, self.beta_err_scale_noise = self.model_accuracy(n_samples=[10000]*10, p_samples=[100]*10, sparsity=[0]*10, sigma=np.logspace(-3, 2, 6), cov_array=[0]*10) 

        ###Collecting Results
        self.times_dict = {'scale' : self.times_scale, 'sparse' : self.times_sparse, 'corr' : self.times_scale_corr, 'noise' : self.times_scale_noise}
        self.mse_dict = {'scale' : self.mse_scale, 'sparse' : self.mse_sparse, 'corr' : self.mse_scale_corr, 'noise' : self.mse_scale_noise}
        self.beta_err_dict = {'scale' : self.beta_err_scale, 'sparse' : self.beta_err_sparse, 'corr' : self.beta_err_corr, 'noise' : self.beta_err_scale_noise}
        self.params_dict = {'scale' : np.logspace(2, 6, 5), 'sparse' : np.linspace(0,.9,10), 'corr' : np.linspace(0, .9, 10), 'noise' : np.logspace(-3, 2, 6)}
        return self.times_dict

    def plot_times(self):
        for time_key in self.times_dict:
            plt.figure()
            plt.title('Model Fitting Time as a Function of Data Size: ' + time_key)
            plt.ylabel('Time to Fit Model in Seconds')
            plt.xlabel('Number of Data Points')
            plt.plot(self.params_dict[time_key], self.times_dict[time_key])
            plt.show()

    def plot_accuracy(self):
        for mse_key in self.mse_dict:
            plt.figure()
            plt.title('MSE as a Function of Data Size: ' + mse_key)
            plt.ylabel('Mean Squared Error')
            plt.xlabel('Number of Data Points')
            plt.plot(self.params_dict[mse_key], self.mse_dict[mse_key])
            plt.show()

    def plot_beta(self):
        for beta_key in self.beta_err_dict:
            plt.figure()
            plt.title('MSE of the Beta as a Function of Data Size: ' + beta_key)
            plt.ylabel('Mean Squared Error')
            plt.xlabel('Number of Data Points')
            plt.plot(self.params_dict[beta_key], self.beta_err_dict[beta_key])
            plt.show()

    def plot_times_together(self):
        count = 1
        plt.figure()
        for time_key in self.times_dict:
            plt.subplot(2,2,count)
            plt.title('Model Fitting Time as a Function of Data Size: ' + time_key)
            plt.ylabel('Time to Fit Model in Seconds')
            plt.xlabel('Number of Data Points')
            plt.plot(self.params_dict[time_key], self.times_dict[time_key])
            count += 1
        plt.show()

    def plot_accuracy_together(self):
        for mse_key in self.mse_dict:
            plt.figure()
            plt.title('MSE as a Function of Data Size: ' + mse_key)
            plt.ylabel('Mean Squared Error')
            plt.xlabel('Number of Data Points')
            plt.plot(self.params_dict[mse_key], self.mse_dict[mse_key])
            plt.show()        
