import numpy as np
import pandas as pd
from sklearn import decomposition
from cycler import cycler
import matplotlib.pyplot as plt
from base_models import DimensionalityReduction
from sklearn import linear_model
import matplotlib.pyplot as plt

class FactorAnalysis(DimensionalityReduction):
    """Class for dimensionality reduction by Factor Analysis, Principal Axis Factoring Method"""

    def __init__(self, intercept=False, scale=False, n_factors=1, **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.n_factors      = n_factors
        self.kwargs         = kwargs

    def _estimate_model(self):
        self.communalities                    = self.EstimateCommunalities()
        self.eigenvalues, self.loading_matrix = self.Eigendecomposition()
        return self.loading_matrix ##In this case, model is defined as loading matrix

    def _estimate_fittedvalues(self):
        fitted_values = self.transform(self.x_train)
        return fitted_values

    def _add_intercept(self, data):
        return data

    def transform(self, data_df):
        factor_scores = np.dot(np.linalg.inv(np.corrcoef(data_df.T)),self.loading_matrix)
        data_demean = data_df - data_df.mean(axis=0)
        data_std = data_demean/data_demean.std()
        factor_df = pd.DataFrame(np.dot(data_std,factor_scores), index=data_std.index, columns=self.loading_matrix.columns)
        return factor_df

    def diagnostics(self):
        super(FactorAnalysis, self).diagnostics() 
        self.scree_plot()

    def EstimateCommunalities(self):
        ###Estimates communalities (r-squared of regression)
        model = linear_model.LinearRegression(fit_intercept=True)
        score_dict = {}
        score_list = []
        
        for column in self.x_train:
            y_var = self.x_train[column]
            x_var = self.x_train.drop([column], axis=1)
            model.fit(x_var, y_var)
            r_squared = model.score(x_var, y_var)
            score_dict[column] = r_squared
            score_list.append(r_squared)
            
        return score_list

    def Eigendecomposition(self):
        corr_mat                    = np.corrcoef(self.x_train.T)
        np.fill_diagonal(corr_mat, self.communalities)
        corr_df                     = pd.DataFrame(corr_mat)
        eigenvalues, eigenvectors   = np.linalg.eig(corr_df)
        if any(x <= 0 for x in eigenvalues[:self.n_factors]): #Checking that there are enough positive eigenvalues
            raise ValueError('There are not enough positive eigenvalues for the number of factors you selected.') 
        loading_matrix = (eigenvectors*np.sqrt(eigenvalues))[:,:self.n_factors]
        factor_str = ['Component']*self.n_factors
        factor_nums = np.linspace(1, self.n_factors, self.n_factors)
        factor_columns = ["%s%02d" % t for t in zip(factor_str, factor_nums)]
        loading_matrix_df = pd.DataFrame(loading_matrix, index=self.x_train.columns, columns=factor_columns)
        return eigenvalues, loading_matrix_df

    def scree_plot(self):
        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.axes([.2, .2, .7, .7])
        plt.plot(self.eigenvalues, linewidth=2)
        plt.axis('tight')
        plt.xlabel('n_components')
        plt.ylabel('Eigenvalues')

    def UpdateCommunalities():
        communalities_update = self.loading_matrix.pow(2).sum(axis=1)
        ###Then call eigendecomposition to update loading matrix. Do many times before calling create factors to finish
