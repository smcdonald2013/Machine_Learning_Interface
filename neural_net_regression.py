import theano
theano.config.gcc.cxxflags='-march=core2'
import pandas as pd
import keras
import abc
from base_models import Regression

class NeuralNetRegression(Regression): #MLP
    def __init__(self, intercept=False, scale=False, batch_size=25, n_epoch=10, loss='mean_squared_error', model_provided=None):
        self.intercept = intercept
        self.scale = scale
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.loss = loss
        self.model_provided = model_provided

    def _estimate_model(self):
        #If user provides model, use that. Otherwise, create default Bayesian Model
        if self.model_provided is not None:
            self.model = self.model_provided
        else:
            #Model creation
            self.model = keras.models.Sequential()
            self.model.add(keras.layers.Dense(100, input_shape=(self.number_feat,)))
            self.model.add(keras.layers.Activation('tanh'))
            self.model.add(keras.layers.Dropout(.5))
            self.model.add(keras.layers.Dense(100))
            self.model.add(keras.layers.Activation('tanh'))
            self.model.add(keras.layers.Dropout(.5))
            self.model.add(keras.layers.Dense(1))
            self.model.add(keras.layers.Activation('linear'))

            self.model.compile(loss=self.loss, optimizer='sgd')

        self.model.fit(self.x_train.values, self.y_train.values, batch_size=self.batch_size, nb_epoch=self.n_epoch, verbose=2) #verbose 2 is common
        return self.model

    def predict(self, x_val):
        super(NeuralNetRegression, self).predict(x_val) 
        val_pred = self.model.predict(self.x_val.values)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred.reshape(x_val.shape[0],), name='predictions')
        return val_df

    def _estimate_fittedvalues(self):
        fittedvals = self.model.predict(self.x_train.values)
        fitted_df   = pd.Series(index=self.x_train.index, data=fittedvals.reshape(self.number_obs,), name='fitted')
        return fitted_df

    def _add_intercept(self, data):
        return data