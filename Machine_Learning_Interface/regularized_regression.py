import numpy as np
import pandas as pd
from sklearn import linear_model
from cycler import cycler
import matplotlib.pyplot as plt
import abc
from .base_models import Regression
from . import scikit_mixin
import warnings
warnings.filterwarnings('ignore')

class RegularizedRegression(Regression):
    """Abstract base class for regularized regression models. 
    Do not create instances of this class for modeling!
    Use derived classes. Note that all derived classes should
    contain the attributes listed.

    Attributes
    ----------
    intercept : boolean
        Whether to fit an intercept to the model.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    cv_folds : int
        Number of folds for k-fold cross validation
    solver : str
        Solver to be used by sklearn, dependent on underlying model fit.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, intercept=False, scale=False, cv_folds=None, solver="Coordinate Descent", **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.cv_folds       = cv_folds
        self.solver         = solver
        self.kwargs         = kwargs

    def diagnostics(self):
        """Regularized regression diagnostics. Includes validation curve, learning curve, and regularization plot."""
        super(RegularizedRegression, self).diagnostics() 
        #self.alphas = np.logspace(-10, 5, 100)
        self.coefs  = self._estimate_coefficients()
        print(self.coefs)
        if self.cv_folds is not None:
            self.alphas = self.model.alphas
            self._gen_cv_paths(self.alphas)
            self.underlying.set_params(alpha=self.model.alpha_)
            scikit_mixin.validation_plot(estimator=self.underlying, title='Validation Plot: Alpha', X=self.x_train, y=self.y_train, cv=5, scoring='mean_squared_error', param_name='alpha', param_range=self.alphas, cv_param=self.model.alpha_)
            scikit_mixin.learning_curve_plot(estimator=self.underlying, title='Learning Curve', X=self.x_train, y=self.y_train, cv=5, scoring='mean_squared_error')
            self.regularization_plot()

    def predict(self, x_val):
        """Prediction using fitted model.

        Parameters
        ----------
        x_val : pd.DataFrame (n_samples, n_features)
            X data for making predictions.

        Returns
        -------
        val_df : pd.Series (n_samples, )
            Predicted values.
        """
        super(RegularizedRegression, self).predict(x_val) 
        val_pred = self.model.predict(self.x_val)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df    

    def _estimate_coefficients(self):
        """Estimates regression coefficients.

        Returns
        -------
        coef_df : pd.Series (n_features, )
            Fitted coefficients of the model.
        """
        coef_vals =  np.append(self.model.coef_,self.model.intercept_)
        coef_names = np.append(self.x_train.columns, 'intercept')
        coef_df = pd.Series(data=coef_vals, index=coef_names, name = 'coefficients')
        return coef_df

    def _estimate_fittedvalues(self):
        """Estimate fitted values.

        Returns
        -------
        fitted_values : pd.Series (n_samples, )
            Fitted values of model.
        """
        fitted_values = self.predict(self.x_train)
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

    def regularization_plot(self):
        """Regularization plot of coffiecients vs regularization parameter.

        Returns
        -------
        plt : matplotlib figure.
            Regularization plot.
        """
        plt.figure()
        plt.set_prop_cycle = cycler(color=['b', 'r', 'g', 'c', 'k', 'y', 'm'])
        plt.semilogx(self.alphas, self.coefs_cv)
        plt.gca().invert_xaxis()
        plt.axvline(self.model.alpha_, linestyle='--', color='k', label='Alpha: CV estimate')
        plt.xlabel('alpha')
        plt.ylabel('weights')
        plt.title('Coefficients as a function of the regularization')
        plt.axis('tight')
        #plt.show()
        return plt

    @abc.abstractmethod
    def _estimate_model(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _gen_cv_paths(self, alphas):
        raise NotImplementedError()

class LassoRegression(RegularizedRegression):
    """Fits Lasso regression using sklearn implementation."""

    def _estimate_model(self):
        """Estimates lasso regression object.

        Returns
        -------
        model : sklearn lasso regression or lasso cv object
            Fitted lasso model.
        """
        ###Lars Algorithm
        if self.solver == "Lars":
            self.underlying = linear_model.LassoLars(fit_intercept=self.intercept, normalize=False)
            if self.cv_folds is 'IC': #For AIC/BIC. criterion kwarg should be provided.  
                model = linear_model.LassoLarsIC(fit_intercept=self.intercept, normalize=False, **self.kwargs)
            elif self.cv_folds is not None:
                model = linear_model.LassoLarsCV(fit_intercept=self.intercept, cv=self.cv_folds, normalize=False, **self.kwargs)
            else:
                model = linear_model.Lasso(fit_intercept=self.intercept, **self.kwargs)
        ###Coordinate Descent Algorithm
        elif self.solver == "Coordinate Descent":
            self.underlying = linear_model.Lasso(fit_intercept=self.intercept)
            if self.cv_folds is not None: 
                model = linear_model.LassoCV(fit_intercept=self.intercept, cv=self.cv_folds, **self.kwargs)
            else:
                model = linear_model.Lasso(fit_intercept=self.intercept, **self.kwargs)
        else:
            raise NotImplementedError('Solver not implemented. Choices are Lars or Coordinate Descent.')
        #self.model.fit(np.asanyarray(self.x_train.values,order='F'), self.y_train)
        model.fit(self.x_train, self.y_train)
        return model

    def _gen_cv_paths(self, alphas):
        """Helper function to generate lasso paths."""
        self.alphas, self.coefs_cv, _ = linear_model.lasso_path(self.x_train, self.y_train, fit_intercept=self.intercept, alphas=alphas)
        self.coefs_cv = self.coefs_cv.T
        
class RidgeRegression(RegularizedRegression):
    """Fits Ridge regression using sklearn implementation."""

    def _estimate_model(self):
        """Estimates ridge regression model.

        Returns
        -------
        model : sklearn ridge regression or ridge cv object
            Fitted ridge model.
        """
        self.underlying = linear_model.Ridge(fit_intercept=self.intercept)
        if (self.cv_folds is not None) or (self.solver in ['svd', 'eigen']): 
            #Ridge CV by default tests a very limited set of alphas, we expand on this 
            alphas = np.logspace(-10, 5, 100)
            model = linear_model.RidgeCV(alphas=alphas, cv=self.cv_folds, fit_intercept=self.intercept, gcv_mode=self.solver, **self.kwargs)
        else:
            model = linear_model.Ridge(fit_intercept=self.intercept, **self.kwargs)
        model.fit(self.x_train, self.y_train)
        return model

    def _gen_cv_paths(self, alphas):
        """Helper function to generate cv paths. """
        self.coefs_cv = []
        for a in alphas:
            self.underlying.set_params(alpha=a)
            self.underlying.fit(self.x_train, self.y_train)
            self.coefs_cv.append(self.underlying.coef_)
            
class ElasticNetRegression(RegularizedRegression):
    """Fits Elastic Net Regression using sklearn implementation."""

    def _estimate_model(self):
        """Estimates elastic net regression model.

        Returns
        -------
        model : sklearn elasticnet or elasticnet cv object
            Fitted elastic net model.
        """
        self.underlying = linear_model.ElasticNet(fit_intercept=self.intercept)
        if self.cv_folds is not None: 
            model = linear_model.ElasticNetCV(fit_intercept=self.intercept, cv=self.cv_folds, **self.kwargs)
        else:
            model = linear_model.ElasticNet(fit_intercept=self.intercept, **self.kwargs)
        model.fit(self.x_train, self.y_train)
        return model

    def _gen_cv_paths(self, alphas):
        """Helper function to generate cv paths."""
        self.alphas, self.coefs_cv, _ = linear_model.enet_path(self.x_train, self.y_train, fit_intercept=self.intercept, alphas=alphas)
        self.coefs_cv = self.coefs_cv.T

    def diagnostics(self):
        """Diagnostics for elastic net regression."""
        super(ElasticNetRegression, self).diagnostics()
        if self.cv_folds is not None:
            self.l1_ratios = [.1, .25, .5, .75, .9, .95, .99]
            self.underlying.set_params(alpha=self.model.alpha_, l1_ratio = self.model.l1_ratio_)
            scikit_mixin.validation_plot(estimator=self.underlying, title='Validation Plot: L1 Ratio', X=self.x_train, y=self.y_train, cv=5, scoring='mean_squared_error', param_name='l1_ratio', param_range=self.l1_ratios, cv_param=self.model.l1_ratio_, scale='linear')
