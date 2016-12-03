import numpy as np
import pandas as pd 
from sklearn import grid_search, learning_curve, svm
import matplotlib.pyplot as plt
from base_models import Regression
import scikit_mixin

class SVR(Regression):
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
    """
    def __init__(self, intercept=False, scale=False, kernel='rbf', parameters=None, cv_folds=None, score=None, **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.kernel         = kernel
        self.parameters     = parameters
        self.cv_folds       = cv_folds
        self.score          = score
        self.kwargs         = kwargs 

    def _estimate_model(self): 
        self.underlying = svm.SVR(kernel=self.kernel, **self.kwargs)
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
        super(SVR, self).diagnostics() 
        self.coefs  = self._estimate_coefficients()
        if self.cv_folds is not None:
            self.cv_params = self.model.best_params_
            self.grid_scores = self.model.grid_scores_
            scikit_mixin.validation_plot(estimator=self.underlying, title='SVM Validation Plot', X=self.x_train, y=self.y_train, cv=5, scoring='mean_squared_error', param_name='C', param_range=self.parameters[0]['C'], cv_param=self.model.best_params_['C'])
            scikit_mixin.learning_curve_plot(estimator=self.underlying, title='SVM Learning Curve', X=self.x_train, y=self.y_train, cv=5, scoring='mean_squared_error')

    def predict(self, x_val):
        super(SVR, self).predict(x_val) 
        val_pred = self.model.predict(self.x_val)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df    
