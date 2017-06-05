import numpy as np
from sklearn import model_selection
from cycler import cycler
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score

def learning_curve_plot(estimator, title, X, y, cv=5, scoring='mean_squared_error', train_sizes=np.linspace(.1, 1.0, 5), **kwargs):
    """Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 5).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects
    """
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    train_sizes, train_scores, test_scores = model_selection.learning_curve(estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring, **kwargs)
    if scoring=='mean_squared_error':
        train_scores = -train_scores
        test_scores = -test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    scores_max = np.max([test_scores, train_scores])
    scores_min = np.min([train_scores, test_scores])
    plt.ylim(.9*scores_min,1.1*scores_max)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score" )
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def validation_plot(estimator, title, X, y, param_name, param_range, cv_param, cv=5, scoring='mean_squared_error', scale='log', **kwargs):
    train_scores, test_scores = model_selection.validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring, **kwargs)
    if scoring=='mean_squared_error':
        train_scores = -train_scores
        test_scores = -test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    scores_max = np.max([train_scores, test_scores])
    scores_min = np.min([train_scores, test_scores])

    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    plt.ylim(.9*scores_min,1.1*scores_max)
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.xscale(scale)
    plt.axvline(cv_param, linestyle='--', color='k', label=param_name + ': CV estimate')
    plt.legend(loc="best")
    return plt

def residual_plot(residuals, fittedvalues):
    plt.scatter(residuals, fittedvalues)
    plt.xlabel('Residuals')
    plt.ylabel('Fitted Values')
    plt.title('Fitted vs Residuals')
    return plt

def qq_plot(residuals):
    fig = sm.qqplot(residuals, line='s')
    return fig

def roc_curve_plot(trueclasses, fittedvalues):
    """Generate ROC curve.

    Parameters
    ----------
    trueclasses : pd.Series
        True classes.

    fittedvalues : pd.Series
        Estimated classes (must be discrete, corresponding to class labels, not probabilities)

    Returns
    -------
    plt : Matplotlib figure object.
        ROC plot.
    """
    fpr, tpr, _ = roc_curve(trueclasses, fittedvalues)
    roc_auc     = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    return plt

def confusion_matrix_plot(truevalues, fittedvalues, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Generate confusion matrix table, for use in classification modeling diagnostics.

    Parameters
    ----------
    truevalues : pd.Series
        True classes.

    fittedvalues : pd.Series
        Estimated classes (must be discrete, corresponding to class labels, not probabilities)

    normalize : boolean
        Whether to normalize the confusion matrix (represent as percentages rather than counts)

    title : str
        Title to give the plot.

    cmap : Matplotlib colormap
        Colormap to use for table coloring.

    Returns
    -------
    plt : Matplotlib figure object.
        Confusion matrix table.
    """
    cm = confusion_matrix(truevalues, fittedvalues)
    classes = set(truevalues.values)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt

def biplot(pcs, evals, evecs, var_names=None, var=True, xpc=1, ypc=2):
    """Biplot, for use in eigendecomposition/SVD techniques.

    Parameters
    ----------
    pcs : pd.DataFrame
        The data presented in principal component space.

    evals : np.array
        Eigenvalues

    evecs : np.array
        Eigenvectors

    var_names : list
        List of names of the variables.

    var : boolean
        Whether to plot the loading matrix weights for the original variables.

    xpc : int
        The index of the principal component to put on the x-axis.

    ypc : int
        The index of the principal component to put on the y-axis.

    Returns
    -------
    plt : Matplotlib figure object.
        Biplot.
    """
    xpc, ypc = (xpc - 1, ypc - 1)
    singvals = np.sqrt(evals)

    xs = pcs.ix[:, xpc] * singvals[xpc]
    ys = pcs.ix[:, ypc] * singvals[ypc]

    plt.figure()
    plt.title('Biplot')
    plt.scatter(xs, ys, c='k', marker='.')

    if var is False: 
        tvars = np.dot(np.eye(pcs.shape[0], pcs.shape[1]), evecs) * singvals

        for i, col in enumerate(var_names):
            x, y = tvars[i][xpc], tvars[i][ypc]
            plt.arrow(0, 0, x, y, color='r', width=0.002, head_width=0.05)
            plt.text(x* 1.4, y * 1.4, col, color='r', ha='center', va='center')
    plt.xlabel('PC{}'.format(xpc + 1))
    plt.ylabel('PC{}'.format(ypc + 1))   
    return plt 

def plot_calibration_curve(est, name, X_train, y_train):
    """Generate a plot fo the calibration curve, for use in classification modeling diagnostics.

    Parameters
    ----------
    est : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    name : string
        Name of the classifier, i.e. "Logistic Regression, SVC, etc".

    X_train : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y_train : array-like, shape (n_samples) or (n_samples, n_features)
        Target relative to X for classification.
    """
    X_test = X_train
    y_test = y_train
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y_test.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    #plt.tight_layout()
    return plt

def class_plot(mod, X, y, y_pred):
    splot = plt.figure()

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    alpha = 0.5

    # class 0: dots
    plt.plot(X0_tp.iloc[:, 0], X0_tp.iloc[:, 1], 'o', alpha=alpha,
             color='red')
    plt.plot(X0_fp.iloc[:, 0], X0_fp.iloc[:, 1], '*', alpha=alpha,
             color='#990000')  # dark red

    # class 1: dots
    plt.plot(X1_tp.iloc[:, 0], X1_tp.iloc[:, 1], 'o', alpha=alpha,
             color='blue')
    plt.plot(X1_fp.iloc[:, 0], X1_fp.iloc[:, 1], '*', alpha=alpha,
             color='#000099')  # dark blue

    # means
    plt.plot(mod.model.means_[0][0], mod.model.means_[0][1],
             'o', color='black', markersize=10)
    plt.plot(mod.model.means_[1][0], mod.model.means_[1][1],
             'o', color='black', markersize=10)

    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])

    return plt

def plot_classes(data, class_labels, feat=(0,1), title="Classes on Space Spanned by Features"):
    feat_1, feat_2 = feat
    n_clusters = class_labels.unique().shape[0]
    colors = cm.spectral(class_labels.astype(float) / n_clusters)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(title)
    ax1.set_xlabel("Feature %s" % data.columns[feat_1])
    ax1.set_ylabel("Feature %s" % data.columns[feat_2])
    ax1.scatter(data.iloc[:, feat_1], data.iloc[:, feat_2], marker='.', s=30, lw=0, alpha=0.7, c=colors)
    plt.show()

def plot_clusters(data, cluster_labels, cluster_centers, feat=(0,1)):
    """Generate a plot of the estimated clusters along with centers (for kmeans).

    Parameters
    ----------
    data : pd.DataFrame
        Data used to fit the model.

    cluster_labels : list
        The labels assigned by the clustering model.

    cluster_centers : pd.DataFrame
        The centers of the estimated clusters.

    feat : tuple
        Tuple containing the indices of the data to plot.
    """
    feat_1, feat_2 = feat
    n_clusters = cluster_centers.shape[0]
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("Clusters on Space Spanned by Features")
    ax1.set_xlabel("Feature %s" % data.columns[feat_1])
    ax1.set_ylabel("Feature %s" % data.columns[feat_2])
    ax1.scatter(data.iloc[:, feat_1], data.iloc[:, feat_2], marker='.', s=30, lw=0, alpha=0.7, c=colors)
    ax1.scatter(cluster_centers.iloc[:, feat_1], cluster_centers.iloc[:, feat_2], marker='o', c="white", s=200, alpha=1)
    #colors = cm.spectral(np.unique(cluster_labels).astype(float) / n_clusters)
    for i, c in enumerate(cluster_centers.values):
        #ax1.scatter(c[0], c[1], marker='$%d$', % i, alpha=1, s=50, c=colors[i,:])
        ax1.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)
    plt.show()

def plot_silhouette(data, cluster_labels):
    """Creates silhouette plot of clusters, to help determine if correct k was chosen.

    Parameters
    ----------
    data : pd.DataFrame
        Original data used to train model.

    cluster_labels : np.array
        Array of estimated labels.
    """
    n_clusters = len(np.unique(cluster_labels))
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(10, 5)

    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    silhouette_avg = silhouette_score(data, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()
