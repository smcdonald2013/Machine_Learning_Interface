import numpy as np
import pandas as pd
from sklearn import discriminant_analysis
import statsmodels.api as sm
from .base_models import Classification

class LDA(Classification):
    """Class for Linear Discriminant Analysis Models.

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    prob : boolean
        Whether to return class probabilities.
    """
    def __init__(self, scale=False, intercept=False, prob=False, **kwargs):
        self.scale          = scale
        self.intercept      = intercept
        self.prob           = prob
        self.kwargs         = kwargs

    def _estimate_model(self):
        """Estimates sklearn LDA model.

        Returns
        -------
        model : sklearn LDA object.
            Fitted object.
        """
        model = discriminant_analysis.LinearDiscriminantAnalysis(**self.kwargs)
        model.fit(self.x_train, self.y_train)
        return model

    def _estimate_coefficients(self):
        """Returns fitted coeficients. For logistic regression, these are the 'log odds'.

        Returns
        -------
        coef_df : pd.Series
            Coefficients of the model.
        """
        coef_array =  np.append(self.model.coef_,self.model.intercept_)
        coef_names = np.append(self.x_train.columns, 'intercept')
        coef_df = pd.Series(data=coef_array, index=coef_names, name = 'Coefficients')
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
    
    def _estimate_prob(self):
        """Returns the probability of each class for the training data.

        Returns
        -------
        prob_array : np.array (n_samples, n_classes)
            Array of fitted probbilities.
        """

        prob_array = self.model.predict_proba(self.x_train)
        return prob_array

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
        data_int = data
        return data_int

    def diagnostics(self):
        """Performs diagnostics. """
        super(LDA, self).diagnostics() 
        self.coefs  = self._estimate_coefficients()

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
        super(LDA, self).predict(x_val) 
        val_pred = self.model.predict(self.x_val)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df   