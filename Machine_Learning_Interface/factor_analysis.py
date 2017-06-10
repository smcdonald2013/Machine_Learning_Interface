import numpy as np
import pandas as pd
from base_models import DimensionalityReduction
from sklearn import linear_model
import scikit_mixin
import matplotlib.pyplot as plt

class FactorAnalysis(DimensionalityReduction):
    """Dimensionality reduction by Factor Analysis, Principal Axis Factoring Method.

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model. Ignored.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1.
    n_factors : int
        Number of factors to retain.
    """

    def __init__(self, intercept=False, scale=False, n_factors=1, **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.n_factors      = n_factors

    def _estimate_model(self):
        """Estimates Factor Analysis using Principal Axis Factoring

        We define the model as the loading matrix.

        Returns
        -------
        loading_matrix : pd.DataFrame(n_features, n_factors)
            Loading matrix of the data, aka eigenvectors scaled by the eigenvalues
        """
        self.communalities                    = self.EstimateCommunalities()
        self.evals, self.loading_matrix, self.evecs = self.Eigendecomposition()
        return self.loading_matrix

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

    def transform(self, x_val):
        """Transforms the provided data into factor scores.

        Parameters
        ----------
        x_val : pd.DataFrame(n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        factor_df : pd.DataFrame(n_samples, n_factors)
            Factor scores of the data.
        """
        super(FactorAnalysis, self).transform(x_val) #dot(x, evecs) transforms data
        factor_scores = np.dot(np.linalg.inv(np.corrcoef(self.x_val.T)),self.loading_matrix)
        #data_demean = data_df - data_df.mean(axis=0)
        #data_std = data_demean/data_demean.std()
        factor_df = pd.DataFrame(np.dot(self.x_val,factor_scores), index=self.x_val.index, columns=self.loading_matrix.columns)
        return factor_df

    def diagnostics(self):
        """Diagnositcs for Factor Analysis.

        Generates a scree plot and a biplot of the first 2 principal components.
        """
        super(FactorAnalysis, self).diagnostics()
        print(self.loading_matrix)
        self.scree_plot(self.evals)
        scikit_mixin.biplot(self.x_train, self.evals, self.evecs, var_names=self.x_train.columns, var=False)

    def EstimateCommunalities(self):
        """Initial estimates of communalities.

        The communalities are estimated as the r-squared of a linear regression.

        Returns
        -------
        score_list : list
            List of initial estimates of communalities.
        """
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
        """Performs eigendecomposition of the correlation matrix, adjusted for the estimated communalities.

        Note that because the correlation matrix is adjusted by the estimated communlaties, there is no guarantee that the matrix is
        positive semi-definite, therefore there may be negative eigenvalues.

        Returns
        -------
        eigenvalues : np.array(n_eigenvalues,)
            Eigenvalues of the adjusted correlation matrix.
        loading_matrix_df : pd.DataFrame(n_features, n_factors)
            Loading matrix.
        """
        corr_mat                    = np.corrcoef(self.x_train.T)
        np.fill_diagonal(corr_mat, self.communalities)
        corr_df                     = pd.DataFrame(corr_mat)
        eigenvalues, eigenvectors   = np.linalg.eig(corr_df)
        positive_indices = np.where(eigenvalues>0)
        eigenvalues =  eigenvalues[positive_indices]
        eigenvectors = eigenvectors[:,positive_indices[0]]
        if len(eigenvalues) < self.n_factors: #Checking that there are enough positive eigenvalues
            raise ValueError('There are not enough positive eigenvalues for the number of factors you selected.')
        loading_matrix = (eigenvectors*np.sqrt(eigenvalues))[:,:self.n_factors]
        factor_str = ['Component']*self.n_factors
        factor_nums = np.linspace(1, self.n_factors, self.n_factors)
        factor_columns = ["%s%02d" % t for t in zip(factor_str, factor_nums)]
        loading_matrix_df = pd.DataFrame(loading_matrix, index=self.x_train.columns, columns=factor_columns)
        return eigenvalues, loading_matrix_df, eigenvectors

    def scree_plot(self, evals):
        """Plots scree plot- proportion of variance explained by each factor.

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

    def UpdateCommunalities(self):
        """Not currently implemented"""
        communalities_update = self.loading_matrix.pow(2).sum(axis=1)
        ###Then call eigendecomposition to update loading matrix. Do many times before calling create factors to finish
