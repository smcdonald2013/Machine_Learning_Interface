import numpy as np
import pandas as pd
from sklearn import lda
import statsmodels.api as sm
from base_models import Classification

class LDA(Classification):
    def __init__(self, scale=False, intercept=False, prob=False, **kwargs):
        self.scale          = scale
        self.intercept      = intercept
        self.prob           = prob
        self.kwargs         = kwargs

    def _estimate_model(self):
        model = lda.LDA(**self.kwargs)
        model.fit(self.x_train, self.y_train)
        return model

    def _estimate_coefficients(self):
        coef_array =  np.append(self.model.coef_,self.model.intercept_)
        coef_names = np.append(self.x_train.columns, 'intercept')
        coef_df = pd.Series(data=coef_array, index=coef_names, name = 'Coefficients')
        return coef_df

    def _estimate_fittedvalues(self):
        yhat = self.predict(self.x_train)
        return yhat
    
    def _estimate_prob(self):
        prob_array = self.model.predict_proba(self.x_train)
        return prob_array

    def _add_intercept(self, data):
        data_int = data
        return data_int

    def diagnostics(self):
        super(LDA, self).diagnostics() 
        self.coefs  = self._estimate_coefficients()

    def predict(self, x_val):
        super(LDA, self).predict(x_val) 
        val_pred = self.model.predict(self.x_val)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df   