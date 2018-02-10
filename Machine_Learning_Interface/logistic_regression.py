import numpy as np
import pandas as pd
import sklearn as sk
import statsmodels.api as sm
from .base_models import Classification

class LogisticRegression(Classification):
    """Class for Logistic Regression models. Currently this
    supports l2 and l1 regularization as implemented in sci-kit learn,
    and well as unregularized logistic regression implemented in statsmodels. 

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    cv_folds : int
        Number of folds for cross validation. 
    penalized : boolean
        Model to use if regularization should be applied.
    prob : boolean
        Whether to return class probabilities.
    """
    def __init__(self, intercept=False, scale=False, cv_folds=None, penalized=False, prob=False, **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.cv_folds       = cv_folds
        self.penalized      = penalized
        self.prob           = prob
        self.kwargs         = kwargs

    def _estimate_model(self):
        """Estimates logistic regression model.

        Sklearn only implements regularized logistic regression, so if traditional logistic regression is desired, statsmodels is used.

        Returns
        -------
        model : sklearn or statsmodels logistic regression object.
            Fitted object.
        """
        if self.penalized:
            if self.cv_folds is not None: 
                model = sk.linear_model.LogisticRegressionCV(fit_intercept=self.intercept, cv=self.cv_folds, **self.kwargs)
            else:
                model = sk.linear_model.LogisticRegression(fit_intercept=self.intercept, **self.kwargs)
            model.fit(self.x_train, self.y_train)
        else: 
            self.underlying = sm.GLM(self.y_train, self.x_train, family=sm.families.Binomial())
            model = self.underlying.fit()
        self.underlying = model
        return model

    def _estimate_coefficients(self):
        """Returns fitted coefficients. For logistic regression, these are the 'log odds'.

        Returns
        -------
        coef_df : pd.Series
            Coefficients of the model.
        """
        if self.penalized:
            if self.n_classes == 2:
                coef_array =  np.append(self.model.coef_,self.model.intercept_)
                coef_names = np.append(self.x_train.columns, 'intercept')
                coef_df = pd.Series(data=coef_array, index=coef_names, name='Log Odds')
            else:
                coef_array = np.append(self.model.coef_, self.model.intercept_.reshape(self.n_classes, 1), axis=1)
                coef_names = np.append(self.x_train.columns, 'intercept')
                coef_df = pd.DataFrame(data=coef_array, index=coef_names, columns=self.model.classes_)
        else: 
            coef_array = self.model.params
            coef_names = self.x_train.columns
            coef_df = pd.Series(data=coef_array, index=coef_names, name = 'Log Odds')
        return coef_df

    def _estimate_fittedvalues(self):
        """Estimates fitted values

        Returns
        -------
        yhat : pd.Series
            Predicted values of x_data based on fitted model.
        """
        if self.penalized:
            yhat = self.predict(self.x_train)
        else: 
            yhat = self.model.fittedvalues
            yhat = [1 if x > .5 else 0 for x in yhat]
            yhat = pd.Series(data=yhat, index=self.x_train.index, name = 'Fitted')
        return yhat
    
    def _estimate_prob(self):
        """Returns the probability of each class for the training data.

        Returns
        -------
        prob_array : np.array (n_samples, n_classes)
            Array of fitted probbilities.
        """
        if self.penalized:
            prob_array = self.model.predict_proba(self.x_train)
        else: 
            proba_array = self.model.fittedvalues
            prob_array = pd.Series(data=proba_array, index=self.x_train.index, name = 'Probabilities')
        return prob_array

    def _add_intercept(self, data):
        """If statsmodels is used, intercept must be manually added. sklearn can handle intercept internally.

        Parameters
        ----------
        data : np.array
            Input data.

        Returns
        -------
        data_int : np.array
            Data with intercept added, if needed.
        """
        if self.penalized:
            data_int = data
        else: 
            data_int = sm.add_constant(data)
        return data_int

    def diagnostics(self):
        """Performs diagnostic including validation plot and learning curve plot. """
        super(LogisticRegression, self).diagnostics() 
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
        super(LogisticRegression, self).predict(x_val) 
        if self.penalized:
            val_pred = self.model.predict(self.x_val)
        else: 
            val_pred = self.underlying.predict(params=self.model.params, exog=self.x_val)
            val_pred = [1 if x > .5 else 0 for x in val_pred]
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df   