import numpy as np
import pandas as pd 
from sklearn import svm, model_selection
from .base_models import Regression
from . import scikit_mixin

class SVR(Regression):
    """Class for Support Vector Regression Models, utilizing sklearn which implements LibSVM and LibLinear.

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model. If True, this adds a column of ones to the x_data.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    kernel : string 
        Kernel used for SVR. Options from sklearn are 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed', or callable. 
    parameters : List
        Parameters to be used for cross-validation. Ignored is cv_folds is None. Must be in the grid search form used by sklearn, i.e. 
        parameters = [{'kernel': ['linear', 'rbf'], 'C': [.1, 1, 10], 'epsilon' : [.1, 1, 10]}]
    cv_folds : int        
        Number of folds for cross validation. If None, model is fit on entire dataset.
    score : str
        Sklearn scoring parameter, used in grid search.
    type : str
        Type of SVR to perform. One of 'eps' or 'nu'.
    """
    def __init__(self, intercept=False, scale=True, kernel='rbf', parameters=None, cv_folds=None, score=None, type='eps', **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.kernel         = kernel
        self.parameters     = parameters
        self.cv_folds       = cv_folds
        self.score          = score
        self.type           = type
        self.kwargs         = kwargs 

    def _estimate_model(self):
        """Estimates SVR model.

        Returns
        -------
        model : sklearn LinearSVR or SVR model or grid search cv object
            Fitted object.
        """
        if self.kernel == 'linear':
            self.underlying = svm.LinearSVR(**self.kwargs)
        else:
            if self.type == 'eps':
                self.underlying = svm.SVR(kernel=self.kernel, **self.kwargs)
            elif self.type == 'nu':
                self.underlying = svm.NuSVR(kernel=self.kernel, **self.kwargs)
            else:
                raise NotImplementedError('Type not implemented. Choices are eps or nu.')
        if self.cv_folds is not None:
            model = model_selection.GridSearchCV(self.underlying, self.parameters, cv=self.cv_folds, scoring=self.score)
        else:
            model = self.underlying
        model.fit(self.x_train, self.y_train)
        return model

    def _estimate_coefficients(self):
        """Returns coefficients, only applicable if kernel='linear'

        Returns
        -------
        coef_df : pd.Series
            Coefficients of the model. None if kernel != 'linear'
        """
        if self.kernel=='linear':
            model = self.underlying if self.cv_folds is None else self.model.best_estimator_
            coef_array =  np.append(model.coef_, model.intercept_)
            coef_names = np.append(self.x_train.columns, 'intercept')
            coef_df = pd.Series(data=coef_array, index=coef_names, name = 'coefficients')
        else: 
            coef_df = None
        return coef_df

    def _estimate_fittedvalues(self):
        """Estimates fitted values

        Returns
        -------
        yhat : pd.Series
            Predicted values of x_data based on fitted model.
        """
        yhat = self.predict(self.x_train)
        return yhat

    def diagnostics(self):
        """Performs diagnostic including validation plot and learning curve plot. """
        super(SVR, self).diagnostics()
        self.coefs  = self._estimate_coefficients()
        if self.cv_folds is not None:
            self.cv_params = self.model.best_params_
            self.cv_results = pd.DataFrame(self.model.cv_results_)
            self.underlying = self.model.best_estimator_
            scikit_mixin.validation_plot(estimator=self.underlying, title='SVM Validation Plot', X=self.x_train, y=self.y_train, cv=5, scoring='mean_squared_error', param_name='C', param_range=self.parameters[0]['C'], cv_param=self.model.best_params_['C'])
            scikit_mixin.learning_curve_plot(estimator=self.underlying, title='SVM Learning Curve', X=self.x_train, y=self.y_train, cv=5, scoring='mean_squared_error')

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
        super(SVR, self).predict(x_val) 
        val_pred = self.model.predict(self.x_val)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df    
