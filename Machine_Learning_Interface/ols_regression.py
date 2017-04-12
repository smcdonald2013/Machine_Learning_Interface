import numpy as np
import pandas as pd 
from base_models import Regression
from sklearn import linear_model

class OLSRegression(Regression):
    """OLS Regression Class. 

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    **kwargs : varies
        Keyword arguments to be passed to model fitting function. 

    Attributes
    ----------
    coefs : pd.Series (n_features, )
        Fitted coefficients of the model.
    """
    def __init__(self, intercept=False, scale=False, **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.kwargs         = kwargs

    def _estimate_model(self):
        """Estimates sklearn OLS model.

        Returns
        -------
        model : sklearn OLS object
            Fitted OLS object from sklearn
        """
        model = linear_model.LinearRegression(fit_intercept=self.intercept)
        model.fit(self.x_train, self.y_train)
        return model

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
        super(OLSRegression, self).predict(x_val) 
        val_pred = self.model.predict(self.x_val)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df    

    def diagnostics(self):
        """Performs diagnostics for OLS. Most are inherited from base regression model."""
        super(OLSRegression, self).diagnostics() 
        self.coefs  = self._estimate_coefficients()
        print self.coefs

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