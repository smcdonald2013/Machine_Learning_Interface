import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from cycler import cycler
import matplotlib.pyplot as plt
import abc
from base_models import Regression
import scikit_mixin

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
        return model

    def eda(self):
        self.series_plot(self.x_train)
        self.acf_plot(self.x_train)
        self.pacf_plot(self.x_train)

    def diagnostics(self):
        super(ARMARegression, self).diagnostics() 
        self.residual_time_plot(self.fittedvalues)
        self.acf_plot(self.fittedvalues)
        self.pacf_plot(self.fittedvalues)
        print self.model.params
        return self.output_df(self.fittedvalues)
        #qqplot(self.fittedvalues, line='q', fit=True)
        #sm.graphics.tsa.plot_acf(self.fittedvalues.values.squeeze(), lags=40)
        #sm.graphics.tsa.plot_pacf(self.fittedvalues, lags=40, ax=ax2)

    def predict(self, x_val):
        super(ARMARegression, self).predict(x_val) 
        #val_pred = self.model.predict(self.x_val)
        val_pred = self.model.resid
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df    

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
