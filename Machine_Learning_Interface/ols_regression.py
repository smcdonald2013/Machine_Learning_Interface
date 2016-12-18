import numpy as np
import pandas as pd 
from base_models import Regression
import scikit_mixin
from sklearn import linear_model
import scikit_mixin

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
    self.intercept : boolean
        Whether to fit an intercept to the model.
    self.scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    """
    def __init__(self, intercept=False, scale=False, **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.kwargs         = kwargs

    def _estimate_model(self): 
        self.model = linear_model.LinearRegression(fit_intercept=self.intercept)
        self.model.fit(self.x_train, self.y_train)
        return self.model

    def predict(self, x_val):
        super(OLSRegression, self).predict(x_val) 
        #self.x_val = self._data_preprocess(x_val, rescale=False)
        val_pred = self.model.predict(self.x_val)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df    

    def diagnostics(self):
        super(OLSRegression, self).diagnostics() 
        self.coefs  = self._estimate_coefficients()

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