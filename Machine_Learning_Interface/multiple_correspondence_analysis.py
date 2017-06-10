import numpy as np
import pandas as pd
import mca
from base_models import DimensionalityReduction
import scikit_mixin
import matplotlib.pyplot as plt

class MCA(DimensionalityReduction):
    """Class for dimensionality reduction by MCA.

    Note that sklearn performs covariance PCA by default, setting scale=True will perform correlation PCA.

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model. Since PCA de-means the data, this parameter is ignored.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1. False=covariance PCA, True=Correlation PCA.
    **kwargs : varies
        Keyword arguments to be passed to model fitting function. """

    def __init__(self, intercept=False, scale=False,  n_factors=None, **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.n_factors      = n_factors
        self.kwargs         = kwargs

    def diagnostics(self):
        """Diagnositcs for PCA.

        Generates a scree plot and a biplot of the first 2 principal components.
        """
        super(MCA, self).diagnostics()
        self.scree_plot(self.model.L)
        self.biplot(pd.DataFrame(self.model.fs_r()), pd.DataFrame(self.model.fs_c()), var_names=self.x_train.columns, var=False)
        self.loading_matrix = pd.DataFrame(self.model.fs_c(), index=self.x_train.columns, columns=self.fittedvalues.columns)
        print(self.loading_matrix)

    def transform(self, x_val):
        """Transforms the provided data into factor scores using MCA.

        Parameters
        ----------
        x_val : pd.DataFrame(n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        val_df : pd.DataFrame(n_samples, n_factors)
            (Row) Factor scores of the data.
        """
        super(MCA, self).transform(x_val)
        val_pred = self.model.fs_r(N=self.n_factors)
        n_factors = val_pred.shape[1]
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
        """Fits MCA to input data.

        Returns
        -------
        model : sklearn PCA object
            Fitted PCA model object.
        """
        model = mca.mca(self.x_train, **self.kwargs)
        return model

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
        plt.xlabel('Number of Factors')
        plt.ylabel('Proportion of Inertia')
        return plt

    def biplot(self, fs_r, fs_c, var_names=None, var=True, xpc=1, ypc=2):

        xpc, ypc = (xpc - 1, ypc - 1)

        xs = fs_r.ix[:, xpc]
        ys = fs_r.ix[:, ypc]

        plt.figure()
        plt.title('Biplot')
        plt.scatter(xs, ys, c='k', marker='.')

        if var is False:
            tvars = fs_c

            for i, col in enumerate(var_names):
                x, y = tvars.ix[i, xpc], tvars.ix[i, ypc]
                plt.arrow(0, 0, x, y, color='r', width=0.002, head_width=0.05)
                plt.text(x * 1.4, y * 1.4, col, color='r', ha='center', va='center')
        plt.xlabel('Factor{}'.format(xpc + 1))
        plt.ylabel('Factor{}'.format(ypc + 1))
        return plt
