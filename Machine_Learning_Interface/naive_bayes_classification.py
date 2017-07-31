import pandas as pd
from sklearn import naive_bayes
from .base_models import Classification

class NaiveBayesClassification(Classification):
    """Class for Naive Bayes Models.

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
        """Estimates sklearn Gaussian NB model.

        Returns
        -------
        model : sklearn NB object.
            Fitted object.
        """
        model = naive_bayes.GaussianNB(**self.kwargs)
        model.fit(self.x_train, self.y_train)
        self.underlying = model
        return model

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
            Array of fitted probabilities.
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
        super(NaiveBayesClassification, self).diagnostics() 

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
        super(NaiveBayesClassification, self).predict(x_val) 
        val_pred = self.model.predict(self.x_val)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df   