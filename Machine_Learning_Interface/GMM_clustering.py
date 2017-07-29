import numpy as np
import pandas as pd
from scipy import linalg
from .base_models import DimensionalityReduction
from . import scikit_mixin
from sklearn import mixture
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl

class GMM(DimensionalityReduction):
    """Clustering using Gaussian Mixture ModelsN.

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model. Ignored.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1.
    **kwargs : varies
        Keyword arguments to be passed to model fitting function. """

    def __init__(self, intercept=False, scale=False, **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.kwargs         = kwargs

    def _estimate_model(self):
        """Fits GMM to input data.

        Returns
        -------
        model : sklearn GMM object
            Fitted GMM object.
        """
        model = mixture.GaussianMixture(**self.kwargs)
        model.fit(self.x_train)
        self.labels = model.predict(self.x_train)
        return model

    def diagnostics(self):
        """Diagnostics for GMM.

        Generates a silhouette plot and a biplot of the clusters on the first 2 features.
        """
        super(GMM, self).diagnostics()
        self.plot_results(self.x_train, self.labels, self.model.means_, self.model.covariances_)
        scikit_mixin.plot_silhouette(data=self.x_train, cluster_labels=self.labels)

    def plot_results(X, data, labels, means, covariances, feat=(0,1)):
        """Plots the data with labels assigned and ellipses for cluster mean/covariance.

        Parameters
        ----------
        data : pd.DataFrame
            Input data used for model fitting.
        labels : np.array
            Labels assigned to data.
        means : np.array
            Estimated means of the data.
        covariances : np.array
            Estimated covariance of the data.
        feat : tuple (int, int)
            Tuple containing the indices of the data to plot.
        """
        color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])
        feat_1, feat_2 = feat
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])

            if not np.any(labels == i): #If there aren't any points corresponding to label, skip.
                continue
            plt.scatter(data.iloc[labels == i, feat_1], data.iloc[labels == i, feat_2], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

        plt.title('Gaussian Mixture Model')
        plt.show()

    def transform(self, x_val):
        """This is probably irrelevant for GMM.

        Parameters
        ----------
        x_val : pd.DataFrame(n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        val_df : pd.DataFrame(n_samples, n_factors)
            Factor scores of the data.
        """
        super(GMM, self).transform(x_val)
        val_pred = self.model.transform(self.x_val)
        n_clusters = self.model.n_clusters
        cluster_str = ['Cluster']*n_clusters
        cluster_nums = np.linspace(1, n_clusters, n_clusters)
        cluster_columns = ["%s%02d" % t for t in zip(cluster_str, cluster_nums)]
        val_df   = pd.DataFrame(index=self.x_val.index, data=val_pred, columns=cluster_columns)
        return val_df

    def _estimate_fittedvalues(self):
        """Simple returns the labels, since GMM doesn't transform data.

        Returns
        -------
        fitted_values : pd.DataFrame(n_samples, n_features)
            Cluster labels.
        """
        fitted_values = self.labels
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
