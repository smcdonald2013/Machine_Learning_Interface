import numpy as np
import pandas as pd
from .base_models import DimensionalityReduction
from . import scikit_mixin
from sklearn import cluster
import itertools

class KMeans(DimensionalityReduction):
    """Clustering using KMeans.

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
        """Fits KMeans to input data.

        Returns
        -------
        model : sklearn KMeans object
            Fitted KMeans model object.
        """
        model = cluster.KMeans(**self.kwargs)
        model.fit(self.x_train)
        n_clusters = model.n_clusters
        cluster_str = ['Cluster']*n_clusters
        cluster_nums = np.linspace(1, n_clusters, n_clusters)
        cluster_columns = ["%s%02d" % t for t in zip(cluster_str, cluster_nums)]
        self.centers = pd.DataFrame(model.cluster_centers_, index=cluster_columns, columns=self.x_train.columns)
        self.labels = model.labels_
        self.n_clusters = n_clusters
        return model

    def diagnostics(self, unscaled=True):
        """Diagnositcs for KMeans.

        Parameters
        ----------
        unscaled : boolean
            Whether to plot scaled or unscaled clusters. Only relevant if scale=True.

        Generates a silhouette plot and a biplot of the clusters on the first 2 features.

        """
        super(KMeans, self).diagnostics() 
        if unscaled:
            x_unscaled = pd.DataFrame(self.scaler.inverse_transform(self.x_train), index=self.x_train.index, columns=self.x_train.columns)
            centers_unscaled = pd.DataFrame(self.scaler.inverse_transform(self.centers), index=self.centers.index, columns=self.centers.columns)
            for feat_pair in itertools.combinations(range(0, self.x_train.shape[1]), r=2):
                scikit_mixin.plot_clusters(data=x_unscaled, cluster_labels=self.labels, cluster_centers=centers_unscaled, feat=feat_pair)
        else:
            for feat_pair in itertools.combinations(range(0, self.x_train.shape[1]), r=2):
                scikit_mixin.plot_clusters(data=self.x_train, cluster_labels=self.labels, cluster_centers=self.centers, feat=feat_pair)
        scikit_mixin.plot_silhouette(data=self.x_train, cluster_labels=self.labels)

    def transform(self, x_val):
        """Transforms the provided data into cluster-distance space, as implemented by sklearn.

        Parameters
        ----------
        x_val : pd.DataFrame(n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        val_df : pd.DataFrame(n_samples, n_factors)
            Factor scores of the data.
        """
        super(KMeans, self).transform(x_val) #dot(x, evecs) transforms data
        val_pred = self.model.transform(self.x_val)
        n_clusters = self.model.n_clusters
        cluster_str = ['Cluster']*n_clusters
        cluster_nums = np.linspace(1, n_clusters, n_clusters)
        cluster_columns = ["%s%02d" % t for t in zip(cluster_str, cluster_nums)]
        val_df   = pd.DataFrame(index=self.x_val.index, data=val_pred, columns=cluster_columns)
        return val_df

    def _estimate_fittedvalues(self):
        """Transforms the input data.

        Returns
        -------
        fitted_values : pd.DataFrame(n_samples, n_features)
            Cluster-distance space representation of the input data.
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
