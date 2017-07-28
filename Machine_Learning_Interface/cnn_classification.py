import theano
#theano.config.gcc.cxxflags='-march=core2'
import pandas as pd
import numpy as np
import keras
import abc
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from .base_models import Classification

class CNNClassification(Classification): #Basic CNN
    def __init__(self, intercept=False, scale=False, prob=False, batch_size=25, n_epoch=10, train_size=.8, loss='categorical_crossentropy', nb_filters=32, pool_size = (2,2), kernel_size = (3,3), model_provided=None):
        self.intercept = intercept
        self.scale = scale
        self.prob = prob
        self.batch_size = batch_size
        self.train_size = train_size
        self.n_epoch = n_epoch
        self.loss = loss
        self.nb_filters = nb_filters
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.model_provided = model_provided

    def _estimate_model(self):
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, train_size=self.train_size)

            #keras parameters
            nb_classes = int(np.unique(self.y_train).shape[0])
            color_dim = int(self.x_train.shape[1])
            img_rows = int(self.x_train.shape[2])
            img_cols = int(self.x_train.shape[3])

            #for theano based input
            input_shape = (color_dim, img_rows, img_cols)

            # convert class vectors to binary class matrices
            Y_train = np_utils.to_categorical(self.y_train.values, nb_classes)
            Y_val = np_utils.to_categorical(self.y_val.values, nb_classes)

            model = Sequential()

            model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1], border_mode='valid', input_shape=input_shape, dim_ordering='th', activation='relu'))
            model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1], dim_ordering='th', activation='relu'))
            model.add(MaxPooling2D(pool_size=self.pool_size,dim_ordering='th'))
            model.add(Dropout(0.25))

            model.add(Flatten(name='feature_layer'))
            model.add(Dense(128))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(nb_classes))
            model.add(Activation('softmax'))

            model.compile(loss=self.loss, optimizer='adadelta', metrics=['accuracy'])

            model.fit(self.x_train, Y_train, batch_size=self.batch_size, nb_epoch=self.n_epoch, validation_data=(self.x_val, Y_val), verbose=1)

            return model

    def predict(self, x_val):
        super(CNNClassification, self).predict(x_val) 
        val_pred = self.model.predict(self.x_val)
        #val_df   = pd.Series(index=self.x_val.index, data=val_pred.reshape(x_val.shape[0],), name='predictions')
        return val_pred

    def _estimate_fittedvalues(self):
        fittedvals = self.model.predict_classes(self.x_train)
        #fitted_df   = pd.Series(index=self.x_train.index, data=fittedvals.reshape(self.number_obs,), name='fitted')
        return fittedvals

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
        if self.scale:
            x_train = self._scale_data(x_train, rescale)
        return x_train