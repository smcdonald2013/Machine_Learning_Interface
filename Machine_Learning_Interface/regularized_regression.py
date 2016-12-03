import numpy as np
import pandas as pd
from sklearn import linear_model, learning_curve
from cycler import cycler
import matplotlib.pyplot as plt
import abc
from base_models import Regression
import scikit_mixin

class RegularizedRegression(Regression):
    """Abstract base class for regularized regression models. 
    Do not create instances of this class for modeling!
    Use derived classes. Note that all derived classes should
    contain the attributes listed.

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    cv_folds : int
        Number of folds for cross validation. If None, no cross-validation is performed. 
    **kwargs : varies
        Keyword arguments to be passed to cross-validation function. Only appropriate if cv_folds is not None

    Attributes
    ----------
    self.intercept : boolean
        Whether to fit an intercept to the model.
    self.scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    self.cv_folds : int        
        Number of folds for cross validation
    self.coefs : pd.Series
        Fitted coefficients. Index is coefficient names. 
        Underlying fitted model. View documentation of derived classes for information
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, intercept=False, scale=False, cv_folds=None, solver="Coordinate Descent", **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.cv_folds       = cv_folds
        self.solver         = solver
        self.kwargs         = kwargs

    def diagnostics(self):
        super(RegularizedRegression, self).diagnostics() 
        self.alphas = np.logspace(-10, 5, 100)
        self.coefs  = self._estimate_coefficients()
        self._gen_cv_paths()
        if self.cv_folds is not None:
            self.underlying.set_params(alpha=self.model.alpha_)
            scikit_mixin.validation_plot(estimator=self.underlying, title='Validation Plot: Alpha', X=self.x_train, y=self.y_train, cv=5, scoring='mean_squared_error', param_name='alpha', param_range=self.alphas, cv_param=self.model.alpha_)
            scikit_mixin.learning_curve_plot(estimator=self.underlying, title='Learning Curve', X=self.x_train, y=self.y_train, cv=5, scoring='mean_squared_error')
        self.regularization_plot()

    def predict(self, x_val):
        super(RegularizedRegression, self).predict(x_val) 
        #self.x_val = self._data_preprocess(x_val, rescale=False)
        val_pred = self.model.predict(self.x_val)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df    

    def _estimate_coefficients(self):
        coef_a =  np.append(self.model.coef_,self.model.intercept_)
        coef_names = np.append(self.x_train.columns, 'intercept')
        coef_df = pd.Series(data=coef_a, index=coef_names, name = 'coefficients')
        return coef_df

    def _estimate_fittedvalues(self):
        fitted_values = self.predict(self.x_train)
        return fitted_values

    def _add_intercept(self, data):
        return data

    def regularization_plot(self):
        plt.figure()
        plt.set_prop_cycle = cycler(color=['b', 'r', 'g', 'c', 'k', 'y', 'm'])
        plt.semilogx(self.alphas, self.coefs_cv)
        plt.gca().invert_xaxis()
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
    def _gen_cv_paths(self):
        raise NotImplementedError()

class LassoRegression(RegularizedRegression):
    def _estimate_model(self):
        ###Lars Algorithm
        if self.solver == "Lars":
            self.underlying = linear_model.LassoLars(fit_intercept=self.intercept, normalize=False)
            if self.cv_folds is 'IC': #For AIC/BIC. criterion kwarg should be provided.  
                self.model = linear_model.LassoLarsIC(fit_intercept=self.intercept, normalize=False, **self.kwargs)
            elif self.cv_folds is not None:
                self.model = linear_model.LassoLarsCV(fit_intercept=self.intercept, cv=self.cv_folds, normalize=False, **self.kwargs)
            else:
                self.model = linear_model.Lasso(fit_intercept=self.intercept, **self.kwargs)
        ###Coordinate Descent Algorithm
        elif self.solver == "Coordinate Descent":
            self.underlying = linear_model.Lasso(fit_intercept=self.intercept)
            if self.cv_folds is not None: 
                self.model = linear_model.LassoCV(fit_intercept=self.intercept, cv=self.cv_folds, **self.kwargs)
            else:
                self.model = linear_model.Lasso(fit_intercept=self.intercept, **self.kwargs)
        else:
            raise NotImplementedError('Solver not implemented. Choices are Lars or Coordinate Descent.')
        #self.model.fit(np.asanyarray(self.x_train.values,order='F'), self.y_train)
        self.model.fit(self.x_train, self.y_train)
        return self.model

    def _gen_cv_paths(self):
        self.alphas, self.coefs_cv, _ = linear_model.lasso_path(self.x_train, self.y_train, fit_intercept=self.intercept, alphas=self.alphas)
        self.coefs_cv = self.coefs_cv.T
        
class RidgeRegression(RegularizedRegression):
    def _estimate_model(self):
        self.underlying = linear_model.Ridge(fit_intercept=self.intercept)
        if (self.cv_folds is not None) or (self.solver in ['svd', 'eigen']): 
            #Ridge CV by default tests a very limited set of alphas, we expand on this 
            alphas = np.logspace(-10, 5, 100)
            self.model = linear_model.RidgeCV(alphas=alphas, cv=self.cv_folds, fit_intercept=self.intercept, gcv_mode=self.solver, **self.kwargs)
        else:
            self.model = linear_model.Ridge(fit_intercept=self.intercept, **self.kwargs)
        self.model.fit(self.x_train, self.y_train)
        return self.model

    def _gen_cv_paths(self):
        self.coefs_cv = []
        for a in self.alphas:
            self.underlying.set_params(alpha=a)
            self.underlying.fit(self.x_train, self.y_train)
            self.coefs_cv.append(self.underlying.coef_)
            
class ElasticNetRegression(RegularizedRegression):
    def _estimate_model(self):
        self.underlying = linear_model.ElasticNet(fit_intercept=self.intercept)
        if self.cv_folds is not None: 
            self.model = linear_model.ElasticNetCV(fit_intercept=self.intercept, cv=self.cv_folds, **self.kwargs)
        else:
            self.model = linear_model.ElasticNet(fit_intercept=self.intercept, **self.kwargs)
        self.model.fit(self.x_train, self.y_train)
        return self.model

    def _gen_cv_paths(self):
        self.alphas, self.coefs_cv, _ = linear_model.enet_path(self.x_train, self.y_train, fit_intercept=self.intercept, alphas=self.alphas)
        self.coefs_cv = self.coefs_cv.T

    def diagnostics(self):
        if self.cv_folds is not None:
            self.l1_ratios = [.1, .25, .5, .75, .9, .95, .99]
            self.underlying.set_params(alpha=self.model.alpha_, l1_ratio = self.model.l1_ratio_)
            scikit_mixin.validation_plot(estimator=self.underlying, title='Validation Plot: L1 Ratio', X=self.x_train, y=self.y_train, cv=5, scoring='mean_squared_error', param_name='l1_ratio', param_range=self.l1_ratios, cv_param=self.model.l1_ratio_, scale='linear')
        super(ElasticNetRegression, self).diagnostics() 
