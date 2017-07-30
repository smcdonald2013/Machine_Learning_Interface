import numpy as np
import pandas as pd 
from sklearn import svm, model_selection
import matplotlib.pyplot as plt
from .base_models import Classification
from . import scikit_mixin

class SVC(Classification):
    """Class for Support Vector Classification Models.

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
    score : str
        Scoring function to be used for cross-validation. Should be one of options from sklearn.
    prob : boolean
        Whether to return class probabilities.
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
        """Estimates sklearn SVC model.

        Returns
        -------
        model : sklearn SVR model or grid search cv object
            Fitted object.
        """
        self.underlying = svm.SVC(kernel=self.kernel, probability=self.prob, **self.kwargs)
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
            coef_array =  np.append(self.model.coef_,self.model.intercept_)
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

    def diagnostics(self):
        """Performs diagnostic including validation plot and learning curve plot. """
        super(SVC, self).diagnostics() 
        self.coefs  = self._estimate_coefficients()
        if self.cv_folds is not None:
            self.cv_params = self.model.best_params_
            self.cv_results = pd.DataFrame(self.model.cv_results_)
            self.underlying = self.model.best_estimator_
            #self.grid_scores = self.model.grid_scores_
            #self.validation_plot()
            #self.plot_calibration_curve(self.underlying, 'SVM Classification', self.x_train, self.y_train, self.x_train, self.y_train)
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
        super(SVC, self).predict(x_val) 
        val_pred = self.model.predict(self.x_val)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df  

    def _estimate_prob(self):
        """Returns the probability of each class for the training data.

        Returns
        -------
        prob_array : np.array (n_samples, n_classes)
            Array of fitted probbilities.
        """
        prob_array = self.model.predict_proba(self.x_train)
        return prob_array