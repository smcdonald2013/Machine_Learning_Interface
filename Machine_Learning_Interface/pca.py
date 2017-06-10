import numpy as np
import pandas as pd
from sklearn import decomposition
from base_models import DimensionalityReduction
import scikit_mixin
import matplotlib.pyplot as plt

class PCA(DimensionalityReduction):
    """Class for dimensionality reduction by PCA.

    Note that sklearn performs covariance PCA by default, setting scale=True will perform correlation PCA.

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model. Since PCA de-means the data, this parameter is ignored.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1. False=covariance PCA, True=Correlation PCA.
    kernel : boolean
        The kernel to be used if kernel PCA is desired. Must be one of options implemented in sklearn.
    **kwargs : varies
        Keyword arguments to be passed to model fitting function. """

    def __init__(self, intercept=False, scale=False, kernel=False, **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.kernel         = kernel
        self.kwargs         = kwargs

    def diagnostics(self):
        """Diagnositcs for PCA.

        Generates a scree plot and a biplot of the first 2 principal components.
        """
        super(PCA, self).diagnostics() 
        self.scree_plot(self.evals)
        scikit_mixin.biplot(self.x_train, self.evals, self.evecs, var_names=self.x_train.columns, var=self.kernel)
        self.loading_matrix = pd.DataFrame(self.evecs, index=self.x_train.columns, columns=self.fittedvalues.columns)
        print(self.loading_matrix)

    def transform(self, x_val):
        """Transforms the provided data into factor scores using the principal components.

        Parameters
        ----------
        x_val : pd.DataFrame(n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        val_df : pd.DataFrame(n_samples, n_factors)
            Factor scores of the data.
        """
        super(PCA, self).transform(x_val) #dot(x, evecs) transforms data
        val_pred = self.model.transform(self.x_val)
        n_factors = self.model.n_components if self.model.n_components is not None else self.x_val.shape[1]
        factor_str = ['Factor']*n_factors
        factor_nums = np.linspace(1, n_factors, n_factors)
        factor_columns = ["%s%02d" % t for t in zip(factor_str, factor_nums)]
        val_df   = pd.DataFrame(index=self.x_val.index, data=val_pred, columns=factor_columns)
        return val_df

    def _estimate_fittedvalues(self):
        """Transforms the input data.

        Returns
        -------
        fitted_values : pd.DataFrame(n_samples, n_features)
            Factor scores of input data.
        """
        fitted_values = self.transform(self.x_train)
        return fitted_values

    def _add_intercept(self, data):
        """Overrides base intercept function, just returning data unchanged since sklearn handles intercept internally.

        Parameters
        ----------
        data : pd.DataFrame (n_samples, n_features)
            Training data.

        Returns
        -------
        data : pd.DataFrame (n_samples, n_features)
            Training data, unchanged.
        """
        return data

    def _estimate_model(self):
        """Fits PCA to input data.

        Returns
        -------
        model : sklearn PCA object
            Fitted PCA model object.
        """
        if self.kernel:
            model = decomposition.KernelPCA(**self.kwargs)
            model.fit(self.x_train)
            self.evecs = model.alphas_
            self.evals = model.lambdas_        
        else: 
            model = decomposition.PCA(**self.kwargs)
            model.fit(self.x_train)
            self.evecs = model.components_.T #dot(evecs, x) gives transformed data
            self.evals = model.explained_variance_
        return model

    def scree_plot(self, evals):
        """Plots scree plot- proportion of variance explained by each principal component.

        Parameters
        ----------
        evals : np.array(n_factors,)

        Returns
        -------
        plt : Matplotlib Plot
            Scree plot object
        """
        ex_var = evals/evals.sum()
        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.axes([.2, .2, .7, .7])
        plt.plot(ex_var, linewidth=2)
        plt.title('Scree Plot')
        plt.axis('tight')
        plt.xlabel('Number of Components')
        plt.ylabel('Proportion of Variance')
        return plt

    def Eigendecomposition(self, n_components=1): #Correlation approach
        """Deprecated."""
        self.n_factors = n_components
        x_stand = (self.x_train - np.mean(self.x_train, axis=0))/np.std(self.x_train, axis=0)
        U, S, V = np.linalg.svd(x_stand, full_matrices=True)

        eigenvalues = S**2/(self.x_train.shape[0])
        eigenvectors = V.T

        if any(x <= 0 for x in eigenvalues[:self.n_factors]): #Checking that there are enough positive eigenvalues
            raise ValueError('There are not enough positive eigenvalues for the number of factors you selected.') 

        loading_matrix = eigenvectors[:,:self.n_factors]
        factor_str = ['Component']*self.n_factors
        factor_nums = np.linspace(1, self.n_factors, self.n_factors)
        factor_columns = ["%s%02d" % t for t in zip(factor_str, factor_nums)]
        loading_matrix_df = pd.DataFrame(loading_matrix, index=self.x_train.columns, columns=factor_columns)

        return eigenvalues, loading_matrix_df
