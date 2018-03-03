import numpy as np
import pandas as pd
from .base_models import DimensionalityReduction
from . import scikit_mixin
from sklearn import cluster
from scipy import linalg
import matplotlib.pyplot as plt
from scipy import sparse
from cycler import cycler
from . import kmeans_clustering
import itertools
import functools

class SpectralClustering(DimensionalityReduction):
    """Clustering using Laplacian Eigenmaps.

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model. Ignored.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1.
    **kwargs : varies
        Keyword arguments to be passed to model fitting function. """

    def __init__(self, intercept=False, scale=False, n_clusters=None, type='random_walk', **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.n_clusters      = n_clusters
        self.type           = type
        self.kwargs         = kwargs

    def unnormalized_laplacian(self, adjacency_mat, k):
        """ Calculates unnormalized laplacian from adjacency matrix.

        Parameters
        ----------
        adjacency_mat : np.array, shape (n_samples, n_samples)
            Adjacency matrix of data.
        k : Int
            Number of clusters to estimate

        Returns
        -------
        laplacian : np.array (n_samples, n_samples)
            Estimated laplacian of data.
        evals : np.array (k, )
            Estimated eigenvalues.
        evecs : np.array (n_samples, k)
            Estimated eigenvectors
        model : Kmeans model object
            Estimated kmeans model.
        """
        if sparse.issparse(adjacency_mat):
            degree_mat = sparse.diags(np.array(adjacency_mat.sum(axis=1).flatten())[0])
            laplacian = degree_mat - adjacency_mat
            evals, evecs = sparse.linalg.eigsh(laplacian, k=k, which='SM')
        else:
            degree_mat = np.diag(adjacency_mat.sum(axis=1))
            laplacian = degree_mat - adjacency_mat
            evals, evecs = linalg.eig(a=laplacian)
            l_idx = evals.argosrt()[0:k]
            evals = evals[l_idx]
            evecs = evecs[:, l_idx]
        model = self.kmeans(evecs, k)
        return laplacian, evals, evecs, model

    def random_walk_laplacian(self, adjacency_mat, k):
        """ Calculates random walk laplacian from adjacency matrix.

        Parameters
        ----------
        adjacency_mat : np.array, shape (n_samples, n_samples)
            Adjacency matrix of data.
        k : Int
            Number of clusters to estimate

        Returns
        -------
        laplacian : np.array (n_samples, n_samples)
            Estimated laplacian of data.
        evals : np.array (k, )
            Estimated eigenvalues.
        evecs : np.array (n_samples, k)
            Estimated eigenvectors
        model : Kmeans model object
            Estimated kmeans model.
        """
        if sparse.issparse(adjacency_mat):
            degrees = np.array(adjacency_mat.sum(axis=1).flatten())[0]
            inverse_degrees = 1.0 / degrees
            inv_degree_mat = sparse.diags(inverse_degrees)
            laplacian = np.dot(inv_degree_mat, adjacency_mat)
            evals, evecs = sparse.linalg.eigsh(laplacian, k=k, which='LM')
        else:
            degree_mat = np.diag(adjacency_mat.sum(axis=1))
            laplacian = np.dot(linalg.inv(degree_mat), adjacency_mat)
            evals, evecs = linalg.eig(a=laplacian)
            l_idx = np.flip(evals.argosrt(), axis=0)[0:k]
            evals = evals[l_idx]
            evecs = evecs[:, l_idx]
        model = self.kmeans(evecs, k)
        return laplacian, evals, evecs, model

    def symmetric_laplacian(self, adjacency_mat, k):
        """ Calculates symmetric laplacian from adjacency matrix.

        Parameters
        ----------
        adjacency_mat : np.array, shape (n_samples, n_samples)
            Adjacency matrix of data.
        k : Int
            Number of clusters to estimate

        Returns
        -------
        laplacian : np.array (n_samples, n_samples)
            Estimated laplacian of data.
        evals : np.array (k, )
            Estimated eigenvalues.
        evecs : np.array (n_samples, k)
            Estimated eigenvectors
        model : Kmeans model object
            Estimated kmeans model.
        """
        if sparse.issparse(adjacency_mat):
            degrees = np.array(adjacency_mat.sum(axis=1).flatten())[0]
            degrees_neg_sqrt = sparse.diags(degrees**(-.5))
            laplacian = functools.reduce(np.dot, [degrees_neg_sqrt, adjacency_mat, degrees_neg_sqrt])
            evals, evecs = sparse.linalg.eigsh(laplacian, k=k, which='LM')
            row_sums = ((evecs ** 2).sum(axis=1)) ** (.5) #Computing row sums for normalization
            evec_norm = evecs / row_sums[:, np.newaxis] #Normalizing the eigenvector matrix
        else:
            degree_mat = np.diag(adjacency_mat.sum(axis=1))
            degrees_neg_sqrt = np.diag(np.diag(degree_mat) ** (-.5))
            laplacian = functools.reduce(np.dot, [degrees_neg_sqrt, adjacency_mat, degrees_neg_sqrt])
            evals, evecs = sparse.linalg.eig(a=laplacian)
            l_idx = np.flip(evals.argsort(), axis=0)[0:k]
            evals = evals[l_idx]
            evecs = evecs[:, l_idx]
            row_sums = ((evecs ** 2).sum(axis=1)) ** (.5) #Computing row sums for normalization
            evec_norm = evecs / row_sums[:, np.newaxis] #Normalizing the eigenvector matrix
        model = self.kmeans(evec_norm, k)
        return laplacian, evals,  evec_norm, model

    def kmeans(self, evecs, k):
        """Implements k-means algorithm, necessary step after e'vecs of laplacian are extracted.

        Parameters
        ----------
        evecs : np.array, shape(n_samples, n_evecs)
        k : int
            Number of clusters.

        Returns
        -------
        model : KMeans Model Object
            Fitted kmeans model.
        """
        model = kmeans_clustering.KMeans(n_clusters=k)
        model.fit(pd.DataFrame(evecs))
        return model

    def _estimate_model(self):
        """Fits SpectralClustering to input data.

        Returns
        -------
        model : sklearn Kmeans object
            Fitted Kmeans model object.
        """
        if type(self.x_train) == pd.core.sparse.frame.SparseDataFrame:
            adj_mat = self.x_train.to_coo()
        elif type(self.x_train) == pd.core.frame.DataFrame:
            adj_mat = self.x_train.values

        if self.type == 'unnormalized':
            output = self.unnormalized_laplacian(adj_mat, self.n_clusters)
        elif self.type == 'random_walk':
            output = self.random_walk_laplacian(adj_mat, self.n_clusters)
        elif self.type == 'symmetric':
            output = self.symmetric_laplacian(adj_mat, self.n_clusters)
        self.laplacian, self.evals, self.evecs, self.model = output
        self.labels = self.model.labels
        self.centers = self.model.centers
        return self.model

    def scree_plot(self, evals):
        """Scree plot: The eigenvalues plotted in descending order.

        Parameters
        ----------
        evals : np.array(n_factors, )

        Returns
        -------
        plt : Matplotlib Plot
            Scree plot object
        """
        ex_var = evals
        plt.figure(1, figsize=(4,3))
        plt.clf()
        plt.axes([.2, .2, .7, .7])
        plt.plot(ex_var, linewidth=2)
        plt.title('Scree Plot')
        plt.axis('tight')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Value')
        return plt

    def plot_evecs(self, evecs, labels):
        """Plot of the eigenvectors of the Laplacian.

        Parameters
        ----------
        evecs : np.array, shape(n_samples, n_evecs)
        labels : List of strings
        """
        plt.autoscale(enable=True, axis='both', tight=None)
        fig = plt.figure()
        for i in range(1, evecs.shape[1]+1):
            ax1 = fig.add_subplot(2, 2, i)
            ax1.set_title("Eigenvector %s" % i)
            round_evecs = np.around(np.sort(evecs[:, i - 1]),5)
            ax1.plot(round_evecs)
        fig.subplots_adjust(hspace=.5)
        plt.show()

    def diagnostics(self):
        """Diagnostics for SpectralClustering.

        Generates a scree plot of evals and plot for evecs.
        """
        super(SpectralClustering, self).diagnostics() 
        self.scree_plot(self.evals)
        self.plot_evecs(self.evecs, labels=self.fittedvalues.columns)
        scikit_mixin.plot_clusters(data=self.fittedvalues, cluster_labels=self.labels, cluster_centers=self.centers)

    def transform(self, x_val):
        """Creates a spectral embedding of the data.

        Parameters
        ----------
        x_val : pd.DataFrame(n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        val_df : pd.DataFrame(n_samples, n_factors)
            Factor scores of the data.
        """
        super(SpectralClustering, self).transform(x_val)
        val_pred = self.evecs
        n_clusters = self.n_clusters
        cluster_str = ['Evec']*n_clusters
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
