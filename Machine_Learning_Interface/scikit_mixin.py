import numpy as np
from sklearn import learning_curve
from cycler import cycler
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

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
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(trueclasses, fittedvalues)
    roc_auc     = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    return plt

def confusion_matrix_plot(truevalues, fittedvalues,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
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
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt

def biplot(pcs, evals, evecs, var_names=None, xpc=1, ypc=2):

    xpc, ypc = (xpc - 1, ypc - 1)
    singvals = np.sqrt(evals)

    xs = pcs.ix[:, xpc] * singvals[xpc]
    ys = pcs.ix[:, ypc] * singvals[ypc]

    plt.figure()
    plt.title('Biplot')
    plt.scatter(xs, ys, c='k', marker='.')

    tvars = np.dot(np.eye(pcs.shape[0], pcs.shape[1]), evecs) * singvals

    for i, col in enumerate(var_names):
        x, y = tvars[i][xpc], tvars[i][ypc]
        plt.arrow(0, 0, x, y, color='r', width=0.002, head_width=0.05)
        plt.text(x* 1.4, y * 1.4, col, color='r', ha='center', va='center')
    plt.xlabel('PC{}'.format(xpc + 1))
    plt.ylabel('PC{}'.format(ypc + 1))   
    return plt 

def plot_calibration_curve(est, name, X_train, y_train):
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

def biplot(lda, X, y, y_pred):
    splot = plt.figure()

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    alpha = 0.5

    # class 0: dots
    plt.plot(X0_tp.ix[:, 0], X0_tp.ix[:, 1], 'o', alpha=alpha,
             color='red')
    plt.plot(X0_fp.ix[:, 0], X0_fp.ix[:, 1], '*', alpha=alpha,
             color='#990000')  # dark red

    # class 1: dots
    plt.plot(X1_tp.ix[:, 0], X1_tp.ix[:, 1], 'o', alpha=alpha,
             color='blue')
    plt.plot(X1_fp.ix[:, 0], X1_fp.ix[:, 1], '*', alpha=alpha,
             color='#000099')  # dark blue

    # means
    plt.plot(lda.model.means_[0][0], lda.model.means_[0][1],
             'o', color='black', markersize=10)
    plt.plot(lda.model.means_[1][0], lda.model.means_[1][1],
             'o', color='black', markersize=10)

    return plot
