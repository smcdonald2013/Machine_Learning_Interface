import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa import stattools
from cycler import cycler
import matplotlib.pyplot as plt
import abc
from base_models import Regression
import scikit_mixin
from sklearn import metrics
from scipy import signal

class ARMARegression(Regression):

    def __init__(self, intercept=False, scale=False, select=True, order=None, cv_folds=None,**kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.select         = select
        self.order          = order
        self.cv_folds       = cv_folds
        self.kwargs         = kwargs

    def _estimate_model(self):
        ###Lars Algorithm
        if self.select:
            info_criterion = sm.tsa.arma_order_select_ic(self.x_train, ic=['aic', 'bic'], trend='nc')
            aic_ind = info_criterion['aic_min_order']
            bic_ind = info_criterion['bic_min_order']
            aic_val = info_criterion['aic'].ix[aic_ind[0], aic_ind[1]]
            bic_val = info_criterion['bic'].ix[bic_ind[0], bic_ind[1]]
            self.order = aic_ind if aic_val < bic_val else bic_ind
            model = sm.tsa.ARMA(self.x_train, self.order).fit()
        else: 
            model = sm.tsa.ARMA(self.x_train, self.order).fit()
        self.k_ar = model.k_ar
        self.y_train = self.y_train[self.k_ar:]
        return model

    def eda(self):
        self.series_plot(self.x_train)
        self.acf_plot(self.x_train)
        self.pacf_plot(self.x_train)
        print self.adf(self.x_train.iloc[:,0])
        self.periodigram(self.y_train)

    def diagnostics(self):
        super(ARMARegression, self).diagnostics() 
        self.residual_time_plot(self.fittedvalues)
        self.acf_plot(self.fittedvalues)
        self.pacf_plot(self.fittedvalues)
        print self.adf(self.y_train)
        print self.model.params
        return self.output_df(self.fittedvalues)
        #qqplot(self.fittedvalues, line='q', fit=True)
        #sm.graphics.tsa.plot_acf(self.fittedvalues.values.squeeze(), lags=40)
        #sm.graphics.tsa.plot_pacf(self.fittedvalues, lags=40, ax=ax2)

    def differencing(self, diff):
        self.diff = diff
        self.x_original = self.x_train
        self.x_train = self.x_train.diff(diff).dropna()
        self.y_train = self.y_train.diff(diff).dropna()
        return self.x_train

    def adf(self, residuals, options = ['nc', 'c', 'ct', 'ctt']):
        adf_dict = {}
        for x in options:
            adf_dict[x] = stattools.adfuller(residuals, maxlag=None, regression=x, autolag='AIC')[1]
        adf_df = pd.DataFrame(adf_dict, index=['P Value'])
        return adf_df

    def predict(self, x_val, start = 1, end = 1, dynamic=True):
        super(ARMARegression, self).predict(x_val) 
        if start and end is not None:
            start = self.number_obs
            end = self.x_val.shape[0]+self.number_obs
        val_pred = self.model.predict(start, end, self.x_val, dynamic)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df    

    def _estimate_coefficients(self):
        coef_a =  np.append(self.model.coef_,self.model.intercept_)
        coef_names = np.append(self.x_train.columns, 'intercept')
        coef_df = pd.Series(data=coef_a, index=coef_names, name = 'coefficients')
        return coef_df

    def _estimate_fittedvalues(self):
        fitted_values = self.predict(self.x_train, start=None, end=None, dynamic=False)[self.k_ar:]
        return fitted_values

    def _add_intercept(self, data):
        return data

    def series_plot(self, series):
        plt.figure()
        series.plot()
        plt.xlabel('Date')
        plt.ylabel(series.columns[0])
        plt.title('Series Plot over Time')
        plt.axis('tight')
        return plt

    def order_select(self, series):
        res = sm.tsa.arma_order_select_ic(series, **self.kwargs)
        return res

    def residual_time_plot(self, residuals):
        plt.figure()
        residuals.plot()
        plt.xlabel('Date')
        plt.ylabel('Residual')
        plt.title('Residual Plot over Time')
        plt.axis('tight')
        return plt

    def acf_plot(self, residuals):
        plt.figure()
        sm.graphics.tsa.plot_acf(residuals.values.squeeze(), lags=40)
        plt.axis('tight')
        return plt

    def pacf_plot(self, residuals):
        plt.figure()
        sm.graphics.tsa.plot_pacf(residuals, lags=40)
        plt.axis('tight')
        return plt

    def output_df(self, residuals):
        aic = self.model.aic
        bic = self.model.bic
        dw = sm.stats.durbin_watson(residuals.values)
        ap = stats.normaltest(residuals)[1]
        output = pd.DataFrame(data=[aic, bic, dw, ap], index=['AIC', 'BIC', 'Durbin-Watson Stat', "D'Agostino-Pearson Normal Test p-value"])
        return output

    def periodigram(self, residuals):
        f, Pxx_den = signal.periodogram(residuals)
        plt.figure()
        plt.semilogy(f, Pxx_den)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()
        return plt

    def fixed_seasonality_test(self, dummies=['D', 'W', 'M', 'Y']):
        return 0

    def dummy(self, x_data):
        dummies = pd.get_dummies(x_data.index.week, prefix='Hello_', drop_first=True)
        dummies_df = pd.DataFrame(dummies.values, index=x_data.index, columns=dummies.columns)
        temp = x_data
        pd.concat([temp,dummies_df], axis=1)


def walk_forward_validation(model, x, y, initial_train, forecast_horizon, walk_forward_period, rolling=True):

    train_start = 0
    train_end = train_start + initial_train
    val_start = train_end + 1
    val_end = val_start + forecast_horizon

    n_periods = y.shape[0]
    score = []

    while val_end <= n_periods:
        x_train = x.iloc[train_start:train_end,:]
        y_train =  y.iloc[train_start:train_end]

        x_val = x.iloc[val_start:val_end,:]
        y_val = y.iloc[val_start:val_end]

        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        err = metrics.mean_squared_error(pred, y_val)
        score.append(err)

        if rolling:
            train_start += walk_forward_period
        train_end += walk_forward_period
        val_start += walk_forward_period
        val_end += walk_forward_period

    return score