import numpy as np
from sklearn import learning_curve
from cycler import cycler
import matplotlib.pyplot as plt

def learning_curve_plot(estimator, title, X, y, cv=5, scoring='mean_squared_error', train_sizes=np.linspace(.1, 1.0, 5), **kwargs):
    """
    Generate a simple plot of the test and traning learning curve.

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
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve.learning_curve(estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring, **kwargs)
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
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def validation_plot(estimator, title, X, y, param_name, param_range, cv_param, cv=5, scoring='mean_squared_error', scale='log', **kwargs):
    train_scores, test_scores = learning_curve.validation_curve(
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
    plt.ylabel('Score')
    plt.ylim(.9*scores_min,1.1*scores_max)
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.xscale(scale)
    plt.axvline(cv_param, linestyle='--', color='k', label=param_name + ': CV estimate')
    plt.legend(loc="best")
    return plt
