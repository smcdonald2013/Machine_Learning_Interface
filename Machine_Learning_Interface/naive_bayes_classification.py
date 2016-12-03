import numpy as np
import pandas as pd
from sklearn import naive_bayes
from base_models import Classification

class NaiveBayesClassification(Classification):
    def __init__(self, scale=False, intercept=False, prob=False, **kwargs):
        self.scale          = scale
        self.intercept      = intercept
        self.prob           = prob
        self.kwargs         = kwargs

    def _estimate_model(self):
        self.model = naive_bayes.GaussianNB(**self.kwargs)
        self.model.fit(self.x_train, self.y_train)
        return self.model

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
        super(NaiveBayesCLassification, self).diagnostics() 

    def predict(self, x_val):
        super(NaiveBayesCLassification, self).predict(x_val) 
        val_pred = self.model.predict(self.x_val)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df   