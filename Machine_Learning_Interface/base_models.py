import numpy as np
import pandas as pd
import sklearn as sk
import abc

class Model(object): 
    """Abstract base class for models.
    Do not create instances of this class for modeling!
    Use derived classes. Note that all derived classes should
    contain the attributes listed.

    Parameters
    ----------
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1

    Attributes
    ----------
    self.scale : boolean
        Whether the data has been scaled to have mean=0 and variance=1
    self.x_train : pd.DataFrame, shape (n_samples, n_features)
        Dataframe containing the x data, after it has gone through any preprocessing
    self.y_train : pd.DataFrame, shape (n_samples, 1)
        Dataframe containing the y data.  
    self.number_obs : int
        Number of observations
    self.number_feat : int
        Number of features
    self.model : varies
        Underlying fitted model. View documentation of derived classes for information
    self.fittedvalues : pd.Series, shape (n_samples, )
        Fitted values of the model
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, scale=False, intercept=False):    
        self.scale                                    = scale
        self.intercept                                = intercept

    def fit(self, x_train, y_train):
        """Fit the model using x_train, y_train as training data.
        This method is not intended to be overridden/exntended by derived classes in general.
        _estimate_model is the method unique to each subclass that should do 
        subclass-specific fitting. 

        Parameters
        ----------
        x_train : pd.DataFrame, shape (n_samples, n_features)
            Training data.
        y_train : pd.DataFrame, shape (n_samples, 1)
            Target values.
        """
        self.x_train                                  = self._data_preprocess(x_train)
        self.y_train                                  = y_train
        self.number_obs                               = self.x_train.shape[0]
        self.number_feat                              = self.x_train.shape[1]

        self.model                                    = self._estimate_model() 

    @abc.abstractmethod
    def diagnostics(self): 
        """Abstract base method for performing modeling diagostics.
        This method should typically be extended in derived classes, 
        rather than completely overridden. 
        """
        self.fittedvalues                             = self._estimate_fittedvalues() 

    @abc.abstractmethod
    def predict(self, x_val): 
        """Abstract base method for making predictions from fitted model.
        This method should typicaly be extended in derived classes, 
        rather than completely overridden. It should be extended to 
        return a pandas Series containing the predictions. 

        Parameters
        ----------
        x_val : pd.DataFrame, shape (n_samples, n_features)
            Validation data to be used for prediction. 
        """
        self.x_val = self._data_preprocess(x_val, rescale=False)

    @abc.abstractmethod
    def _estimate_model(self):
        pass

    @abc.abstractmethod
    def _estimate_fittedvalues(self):
        pass

    def _data_preprocess(self, x_train, rescale=True):
        """Data preprocessing method. Only extended by 
        derived classes if necessary.  

        Parameters
        ----------
        x_train : pd.DataFrame, shape (n_samples, n_features)
            Data to be preprocessed. 
        rescale : Boolean
            Whether to rescale data. Should be True for training data,
            False for testing data. 

        Returns
        ----------
        x_train : pd.DataFrame, shape (n_samples, n_features)
            Preprocessed data. Note that n_features may have
            change

        """
        x_train = x_train.copy(deep=True) #Important! If not included, original dataframe will be altered
        if self.scale:
            x_train = self._scale_data(x_train, rescale)
        if self.intercept:
            x_train = self._add_intercept(x_train)
        return x_train

    def _scale_data(self, data, rescale=True):
        """Method for scaling data.  

        Parameters
        ----------
        x_train : pd.DataFrame, shape (n_samples, n_features)
            Data to be preprocessed. 
        rescale : Boolean
            Whether to rescale data. Should be True for training data,
            False for testing data. 

        Returns
        ----------
        scaled_data_df : pd.DataFrame, shape (n_samples, n_features)
            Scaled data. 

        """
        if rescale:
            self.scaler =   sk.preprocessing.StandardScaler().fit(data)
        scaled_data = self.scaler.transform(data)
        scaled_data_df = pd.DataFrame(data=scaled_data, index=data.index, columns=data.columns)
        return scaled_data_df

    def _add_intercept(self, data):
        predictors = list(data)
        data['const'] = np.ones(len(data))
        data = data[['const'] + predictors]
        return data


class Regression(Model): 
    __metaclass__ = abc.ABCMeta

    def __init__(self, intercept=False, scale=False):    
        self.intercept                                = intercept
        self.scale                                    = scale

    def diagnostics(self): 
        """Abstract base method for performing modeling diagostics.
        This method should typically be extended in derived classes, 
        rather than completely overridden. 
        """
        super(Regression, self).diagnostics() 
        self.resid                                    = self._estimate_residuals()
        self.tss, self.ess                            = self._estimate_tss_ess()
        self.rsquared, self.rsquared_adj              = self._estimate_r2_r2adj()
        self.mse                                      = self._estimate_mse()

    @abc.abstractmethod
    def predict(self, x_val): 
        """Abstract base method for making predictions from fitted model.
        This method should typically be extended in derived classes, 
        rather than completely overridden. It should be extended to 
        return a pandas Series containing the predictions. 
        """
        super(Regression, self).predict(x_val) 

    @abc.abstractmethod
    def _estimate_model(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _estimate_fittedvalues(self):
        raise NotImplementedError()

    def _estimate_residuals(self):
        resids = pd.Series(self.y_train.subtract(self.fittedvalues), name = 'resid')
        return resids

    def _estimate_tss_ess(self):
        #TSS Total Sum of Squres
        tss = (self.y_train.subtract(self.y_train.mean()) ** 2.0).sum()

        # ESS - Explained Sum of Squres
        ess = (self.fittedvalues.subtract(self.y_train.mean()) ** 2.0).sum()
        return tss, ess

    def _estimate_r2_r2adj(self):
        r2 = 1.0 - ((self.resid ** 2.0).sum() / self.tss)
        divisor_sub = 2.0 if self.intercept else 1.0
        r2_adj = 1.0 - (((1.0 - r2) * (self.number_obs - 1.0)) / (self.number_obs - divisor_sub - 1.0))
        return r2, r2_adj

    def _estimate_mse(self):
        mse = sk.metrics.mean_squared_error(self.fittedvalues, self.y_train)
        return mse

class Classification(Model): 
    __metaclass__ = abc.ABCMeta

    def __init__(self, scale=False, intercept=False, prob=False):    
        self.scale              = scale
        self.intercept          = intercept
        self.prob               = prob

    def diagnostics(self): 
        """Abstract base method for performing modeling diagostics.
        This method should typically be extended in derived classes, 
        rather than completely overridden. 
        """
        super(Classification, self).diagnostics() 
        if self.prob:
            self.prob_array = self._estimate_prob()
            self.log_loss = self._estimate_log_loss()
        self.accuracy                                 = self._estimate_accuracy()

    @abc.abstractmethod
    def predict(self, x_val): 
        """Abstract base method for making predictions from fitted model.
        This method should typically be extended in derived classes, 
        rather than completely overridden. It should be extended to 
        return a pandas Series containing the predictions. 
        """
        super(Classification, self).predict(x_val) 

    @abc.abstractmethod
    def _estimate_model(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _estimate_fittedvalues(self):
        raise NotImplementedError()

    def _estimate_log_loss(self):
        log_loss = sk.metrics.log_loss(self.fittedvalues, self.y_train)
        return log_loss

    def _estimate_accuracy(self):
        log_loss = sk.metrics.accuracy_score(self.fittedvalues, self.y_train)
        return log_loss

class DimensionalityReduction(object): 
    """Abstract base class for dimensionality reduction.
    Do not create instances of this class for modeling!
    Use derived classes. Note that all derived classes should
    contain the attributes listed.

    Parameters
    ----------
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1

    Attributes
    ----------
    self.scale : boolean
        Whether the data has been scaled to have mean=0 and variance=1
    self.x_train : pd.DataFrame, shape (n_samples, n_features)
        Dataframe containing the x data, after it has gone through any preprocessing
    self.y_train : pd.DataFrame, shape (n_samples, 1)
        Dataframe containing the y data.  
    self.number_obs : int
        Number of observations
    self.number_feat : int
        Number of features
    self.model : varies
        Underlying fitted model. View documentation of derived classes for information
    self.fittedvalues : pd.Series, shape (n_samples, )
        Fitted values of the model
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, scale=False, intercept=False):    
        self.scale                                    = scale
        self.intercept                                = intercept

    def fit(self, x_train):
        """Fit the model using x_train as training data.
        This method is not intended to be overridden/exntended by derived classes in general.
        _estimate_model is the method unique to each subclass that should do 
        subclass-specific fitting. 

        Parameters
        ----------
        x_train : pd.DataFrame, shape (n_samples, n_features)
            Training data.
        y_train : pd.DataFrame, shape (n_samples, 1)
            Target values.
        """
        self.x_train                                  = self._data_preprocess(x_train)
        self.number_obs                               = self.x_train.shape[0]
        self.number_feat                              = self.x_train.shape[1]

        self.model                                    = self._estimate_model() 

    @abc.abstractmethod
    def diagnostics(self): 
        """Abstract base method for performing modeling diagostics.
        This method should typically be extended in derived classes, 
        rather than completely overridden. 
        """
        self.fittedvalues                             = self._estimate_fittedvalues() 

    @abc.abstractmethod
    def transform(self, x_val): 
        """Abstract base method for transforming data from fitted model.
        This method should typicaly be extended in derived classes, 
        rather than completely overridden. It should be extended to 
        return a pandas Series containing the transformed data. 

        Parameters
        ----------
        x_val : pd.DataFrame, shape (n_samples, n_features)
            Validation data to be used for prediction. 
        """
        self.x_val = self._data_preprocess(x_val, rescale=False)

    @abc.abstractmethod
    def _estimate_model(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _estimate_fittedvalues(self):
        raise NotImplementedError()

    def _data_preprocess(self, x_train, rescale=True):
        """Data preprocessing method. Only extended by 
        derived classes if necessary.  

        Parameters
        ----------
        x_train : pd.DataFrame, shape (n_samples, n_features)
            Data to be preprocessed. 
        rescale : Boolean
            Whether to rescale data. Should be True for training data,
            False for testing data. 

        Returns
        ----------
        x_train : pd.DataFrame, shape (n_samples, n_features)
            Preprocessed data. Note that n_features may have
            change

        """
        x_train = x_train.copy(deep=True) #Important! If not included, original dataframe will be altered
        if self.scale:
            x_train = self._scale_data(x_train, rescale)
        if self.intercept:
            x_train = self._add_intercept(x_train)
        return x_train

    def _scale_data(self, data, rescale=True):
        """Method for scaling data.  

        Parameters
        ----------
        x_train : pd.DataFrame, shape (n_samples, n_features)
            Data to be preprocessed. 
        rescale : Boolean
            Whether to rescale data. Should be True for training data,
            False for testing data. 

        Returns
        ----------
        scaled_data_df : pd.DataFrame, shape (n_samples, n_features)
            Scaled data. 

        """
        if rescale:
            self.scaler =   sk.preprocessing.StandardScaler().fit(data)
        scaled_data = self.scaler.transform(data)
        scaled_data_df = pd.DataFrame(data=scaled_data, index=data.index, columns=data.columns)
        return scaled_data_df

    def _add_intercept(self, data):
        predictors = list(data)
        data['const'] = np.ones(len(data))
        data = data[['const'] + predictors]
        return data