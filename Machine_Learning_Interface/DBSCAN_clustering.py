import numpy as np
import pandas as pd
from base_models import DimensionalityReduction
import scikit_mixin
from sklearn import cluster
from scipy import linalg
import matplotlib.pyplot as plt

class DBSCAN(DimensionalityReduction):
    """Clustering using DBSCAN.

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
        """Fits DBSCAN to input data.

        Returns
        -------
        model : sklearn DBSCAN object
            Fitted DBSCAN model object.
        """
        if type(self.x_train) == pd.core.sparse.frame.SparseDataFrame:
            adj_mat = self.x_train.to_coo()
        elif type(self.x_train) == pd.core.frame.DataFrame:
            adj_mat = self.x_train.values

        model = cluster.DBSCAN(**self.kwargs)
        model.fit(self.x_train)
        self.labels = model.labels_
        return model

    def diagnostics(self, unscaled=False):
        """Diagnostics for DBSCAN.

        Generates a silhouette plot and a biplot of the clusters on the first 2 features.
        """
        super(DBSCAN, self).diagnostics()
        self.output_plot(self.x_train, self.labels, self.model.core_sample_indices_)
        scikit_mixin.plot_silhouette(data=self.x_train, cluster_labels=self.labels)

    def output_plot(data, cluster_labels, core_samples, feat=(0, 1)):
        """Plots the clustering on space spanned by 2 features, distinguishing between core, non-core, and noisy samples.

        Parameters
        ----------
        data : pd.DataFrame
            Data used to fit the model.
        cluster_labels : list
            The labels assigned by the clustering model.
        core_samples : pd.DataFrame
            Indices of points corresponding to core samples
        feat : tuple (int, int)
            Tuple containing the indices of the data to plot.
        """
        feat_1, feat_2 = feat
        unique_labels = set(cluster_labels)
        n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0) #Noise is not a cluster

        colors = plt.cm.spectral(np.linspace(0, 1, len(unique_labels)))
        core_samples_mask = np.zeros_like(cluster_labels, dtype=bool)
        core_samples_mask[core_samples] = True

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = 'k' #Black used for noise

            class_member_mask = (cluster_labels == k)

            xy = data.iloc[class_member_mask & core_samples_mask]
            plt.plot(xy.iloc[:, feat_1], xy.iloc[:, feat_2], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)

            xy = data.iloc[class_member_mask & ~core_samples_mask]
            plt.plot(xy.iloc[:, feat_1], xy.iloc[:, feat_2], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

        plt.title('Estimated Number of Clusters %d' % n_clusters_)
        plt.xlabel("Feature %s" % data.columns[feat_1])
        plt.ylabel("Feature %s" % data.columns[feat_2])
        plt.show()

    def transform(self, x_val):
        """This is probably irrelevant for DBSCAN.

        Parameters
        ----------
        x_val : pd.DataFrame(n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        val_df : pd.DataFrame(n_samples, n_factors)
            Factor scores of the data.
        """
        super(DBSCAN, self).transform(x_val)
        val_pred = self.model.transform(self.x_val)
        n_clusters = self.model.n_clusters
        cluster_str = ['Cluster']*n_clusters
        cluster_nums = np.linspace(1, n_clusters, n_clusters)
        cluster_columns = ["%s%02d" % t for t in zip(cluster_str, cluster_nums)]
        val_df   = pd.DataFrame(index=self.x_val.index, data=val_pred, columns=cluster_columns)
        return val_df

    def _estimate_fittedvalues(self):
        """Simple returns the labels, since DBSCAN doesn't transform data.

        Returns
        -------
        fitted_values : pd.DataFrame(n_samples, n_features)
            Cluster labels.
        """
        fitted_values = labels
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
