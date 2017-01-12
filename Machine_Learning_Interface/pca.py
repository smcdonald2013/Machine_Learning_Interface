import numpy as np
import pandas as pd
from sklearn import decomposition
from cycler import cycler
import matplotlib.pyplot as plt
from base_models import DimensionalityReduction
import scikit_mixin
import matplotlib.pyplot as plt

class PCA(DimensionalityReduction):
    """Class for dimensionality reduction by PCA"""

    def __init__(self, intercept=False, scale=False, **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.kwargs         = kwargs

    def diagnostics(self):
        super(PCA, self).diagnostics() 
        self.scree_plot()
        self.evecs = self.model.components_ #dot(evecs, x) gives transformed data
        self.evals = self.model.explained_variance_
        scikit_mixin.biplot(self.val_df, self.evals, self.evecs, var_names=self.x_val.columns)

    def transform(self, x_val):
        super(PCA, self).transform(x_val) 
        val_pred = self.model.transform(self.x_val)
        val_df   = pd.DataFrame(index=self.x_val.index, data=val_pred)  ###TO DO: Title Factors
        self.val_df = val_df
        return val_df    

    def _estimate_fittedvalues(self):
        fitted_values = self.transform(self.x_train)
        return fitted_values

    def _add_intercept(self, data):
        return data

    def _estimate_model(self):
        model = decomposition.PCA(**self.kwargs)
        model.fit(self.x_train)
        return model

    def scree_plot(self):
        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.axes([.2, .2, .7, .7])
        plt.plot(self.model.explained_variance_ratio_, linewidth=2)
        plt.title('Scree Plot')
        plt.axis('tight')
        plt.xlabel('n_components')
        plt.ylabel('explained_variance_ratio_')
  
    #Correlation or covariance matrix fitting option
    #eigenvalues (scree plot)
    #proportion of variance (scree plot)
    #right eigenvectors
    #Loading matrix?!?!?!
    #Transformed data
    #Unique variances (FA only)

    def Eigendecomposition(self): ###Computes eigendecomposition via SVD (this is the covariance approach to PCA, using centered variables, not standardized)
        x_cen                       = self.x_train - x_train.mean(axis=0)
        U, S, V = linalg.svd(X, full_matrices=True)

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
