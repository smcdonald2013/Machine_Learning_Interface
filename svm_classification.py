import numpy as np
import pandas as pd 
from sklearn import grid_search, learning_curve, svm
import matplotlib.pyplot as plt
from base_models import Classification

class SVC(Classification):
    """Class for Support Vector Regression Models. 

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    kernel : string 
        Kernel used for SVR. Options from sklearn are 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed', or callable. 
    parameters : List
        Parameters to be used for cross-validation. Ignored is cv_folds is None. Must be in the grid search form used by sklearn, i.e. 
        parameters = [{'kernel': ['linear', 'rbf'], 'C': [.1, 1, 10], 'epsilon' : [.1, 1, 10]}]
    cv_folds : int        
        Number of folds for cross validation. If None, model is fit on entire dataset. 

    Attributes
    ----------
    self.intercept : boolean
        Whether to fit an intercept to the model. Ignored if model_provided is not None. 
    self.scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    self.cv_folds : int        
        Number of folds for cross validation. If None, 
    self.niter : int        
        Number of iterations to use when fitting the model. 
    """
    def __init__(self, intercept=False, scale=False, prob=False, kernel='rbf', parameters=None, cv_folds=None, score=None, **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.kernel         = kernel
        self.parameters     = parameters
        self.cv_folds       = cv_folds
        self.score          = score
        self.prob           = prob
        self.kwargs         = kwargs 

    def _estimate_model(self): 
        self.underlying = svm.SVC(kernel=self.kernel, **self.kwargs)
        if self.cv_folds is not None: 
            self.model = grid_search.GridSearchCV(self.underlying, self.parameters, cv=self.cv_folds, scoring=self.score)
        else:
            self.model = self.underlying
        self.model.fit(self.x_train, self.y_train)
        return self.model

    def _estimate_coefficients(self):
        if self.kernel=='linear':
            coef_array =  np.append(self.model.coef_,self.model.intercept_)
            coef_names = np.append(self.x_train.columns, 'intercept')
            coef_df = pd.Series(data=coef_array, index=coef_names, name = 'coefficients')
        else: 
            coef_df = None
        return coef_df

    def _estimate_fittedvalues(self):
        yhat = self.predict(self.x_train)
        return yhat

    def _add_intercept(self, data):
        return data

    def diagnostics(self):
        super(SVC, self).diagnostics() 
        self.coefs  = self._estimate_coefficients()
        if self.cv_folds is not None:
            self.cv_params = self.model.best_params_
            self.grid_scores = self.model.grid_scores_
            self.validation_plot()
            self.plot_calibration_curve(self.underlying, 'SVM Classification', self.x_train, self.y_train, self.x_train, self.y_train)

    def predict(self, x_val):
        super(SVC, self).predict(x_val) 
        val_pred = self.model.predict(self.x_val)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df  

    def _estimate_prob(self):
        prob_array = self.model.predict_proba(self.x_train)
        return prob_array  

    def validation_plot(self):
        param_name = "C"
        param_range = self.parameters[0][param_name]
        train_scores, test_scores = learning_curve.validation_curve(
            self.underlying, self.x_train, self.y_train, param_name=param_name, param_range=param_range,
            cv=5, scoring="mean_squared_error")
        train_scores = -train_scores
        test_scores = -test_scores
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        train_scores_max = np.max(train_scores)
        train_scores_min = np.min(train_scores)

        plt.title('Validation Curve')
        plt.xlabel('Parameter')
        plt.ylabel('Accuracy')
        plt.ylim(train_scores_min,train_scores_max)
        plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="g")
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
        plt.legend(loc="best")
        plt.show()

    def plot_calibration_curve(self, est, name, X, y, X_val, y_val):
        """Plot calibration curve for est w/o and with calibration. """
        # Calibrated with isotonic calibration
        isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

        # Calibrated with sigmoid calibration
        sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

        # Logistic regression with no calibration as baseline
        lr = LogisticRegression(C=1., solver='lbfgs')

        fig = plt.figure()
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for clf, name in [(lr, 'Logistic'),
                          (est, name),
                          (isotonic, name + ' + Isotonic'),
                          (sigmoid, name + ' + Sigmoid')]:
            clf.fit(X, y)
            y_pred = clf.predict(X_val)
            if hasattr(clf, "predict_proba"):
                prob_pos = clf.predict_proba(X_val)[:, 1]
            else:  # use decision function
                prob_pos = clf.decision_function(X_val)
                prob_pos = \
                    (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

            clf_score = brier_score_loss(y_val, prob_pos, pos_label=y.max())
            print("%s:" % name)
            print("\tBrier: %1.3f" % (clf_score))
            print("\tPrecision: %1.3f" % precision_score(y_val, y_pred))
            print("\tRecall: %1.3f" % recall_score(y_val, y_pred))
            print("\tF1: %1.3f\n" % f1_score(y_val, y_pred))

            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_val, prob_pos, n_bins=10)

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

        plt.tight_layout()
        plt.show()