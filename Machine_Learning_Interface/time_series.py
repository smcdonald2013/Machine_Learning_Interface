import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa import stattools
from cycler import cycler
import matplotlib.pyplot as plt
import abc
from .base_models import Regression
from . import scikit_mixin
from sklearn import metrics
from scipy import signal
import calendar
import itertools

class ARMARegression(Regression):
    """Class for ARMA Regression with capability for harmonic regression and detrending.

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1.
    select : boolean
        Whether to select the order using AIC/BIC.
    order : tuple (int, int)
        Order of the AR and MA terms. If select=True this should be None.
    kernel : boolean
        The kernel to be used if kernel PCA is desired. Must be one of options implemented in sklearn.
    **kwargs : varies
        Keyword arguments to be passed to model fitting function. """

    def __init__(self, intercept=False, scale=False, select=True, order=None, **kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.select         = select
        self.order          = order
        self.kwargs         = kwargs

    def _estimate_model(self):
        """Estimates the ARMA regression.

        Returns
        -------
        model : statsmodels ARMA model object
            Fitted ARMA model.
        """
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

    def eda(self, x_data):
        """Performs Exploratory Data Analysis for Time Series Models.

        Parameters
        ----------
        x_data : pd.DataFrame
            Input dataframe. Should be 1 column df.
        """
        self.series_plot(x_data)
        self.acf_plot(x_data)
        self.pacf_plot(x_data)
        print(self.adf(x_data.iloc[:,0]))
        self.periodigram(x_data.iloc[:,0])

    def diagnostics(self):
        """Performs diagnostic tests/plots after model fitting.

        Includes residual plot over time, ACF/PACF, ADF test results, and output df.

        Returns
        ----------
        output_df : pd.DataFrame
            Results for various diagnostics tests.
        """
        super(ARMARegression, self).diagnostics() 
        self.residual_time_plot(self.fittedvalues)
        self.acf_plot(self.fittedvalues)
        self.pacf_plot(self.fittedvalues)
        print(self.adf(self.y_train))
        print(self.model.params)
        return self.output_df(self.fittedvalues)
        #qqplot(self.fittedvalues, line='q', fit=True)
        #sm.graphics.tsa.plot_acf(self.fittedvalues.values.squeeze(), lags=40)
        #sm.graphics.tsa.plot_pacf(self.fittedvalues, lags=40, ax=ax2)

    def differencing(self, diff, x_data=None):
        """Performs differencing on the x_data.

        Note that this changes the model's saved x_data.

        Parameters
        ----------
        diff : int
            Number of differences to take

        Returns
        -------
        x_train : pd.DataFrame
            Differenced dataset.
        """
        self.diff = diff
        if hasattr(self, 'x_train'):
            self.x_original = self.x_train
            self.x_train = self.x_train.diff(diff).dropna()
            self.y_train = self.y_train.diff(diff).dropna()
            diff_data = self.x_train
        else:
            diff_data = x_data.diff(diff).dropna()
        return diff_data

    def adf(self, residuals, options = ['nc', 'c', 'ct', 'ctt']):
        """

        Parameters
        ----------
        residuals : pd.DataFrame
            Dataframe containing the residuals of the model.
        options : list
            Which ADF tests to perform. Options are nc, c, ct, ctt.

        Returns
        -------
        adf_df : pd.DataFrame
            Results of ADF tests.
        """
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

    def eda_regression(self, y_data, x_data, intercept=True):
        if intercept:
            x_data = sm.add_constant(x_data, prepend=False)
        mod = sm.OLS(y_data, x_data)
        res = mod.fit()
        print(res.summary())
        residuals_df = pd.DataFrame(res.resid, columns=[y_data.name])
        return res, residuals_df

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
        max_idx = np.argsort(Pxx_den)[-1]
        max_freq = f[max_idx]
        plt.figure()
        plt.semilogy(f, Pxx_den)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.axvline(max_freq, linestyle='--', color='k', label='Frequency With Max Power: ' + "{0:.3f}".format(max_freq))
        plt.legend(loc='best')
        plt.show()
        return max_freq, plt

    def freq_choice(self,  index, freq):
        if freq == 'Y':
            date_categorical = index.year
        elif freq == 'M':
            # date_categorical = index.month
            month_dict = {v: k for v, k in enumerate(calendar.month_abbr)}
            date_categorical = pd.DataFrame(index.month).replace(month_dict).values.flatten()
        elif freq == 'W':
            date_categorical = index.week
        elif freq == 'D':
            # date_categorical = index.dayofweek
            day_dict = {v: k for v, k in enumerate(calendar.day_abbr)}
            date_categorical = pd.DataFrame(index.dayofweek).replace(day_dict).values.flatten()
        return date_categorical

    def dummy(self, x_data, freq='M'):
        date_categorical = self.freq_choice(x_data.index, freq)
        dummies = pd.get_dummies(date_categorical, prefix='TimeDummy_', drop_first=True)
        dummies_df = pd.DataFrame(dummies.values, index=x_data.index, columns=dummies.columns)
        #temp = x_data
        #final = pd.concat([temp, dummies_df], axis=1)
        #return final
        return dummies_df

    def harmonic_terms(self, x_data, period, n_k=1):
        freq = 1.0/period
        nrows = x_data.shape[0]
        ncols = n_k * 2
        time = pd.Series(range(nrows))
        dta_array = np.empty(shape=[nrows, ncols])
        for k in range(1, n_k + 1):
            sin_dta = np.sin(2 * np.pi * k * freq * time)
            cos_dta = np.cos(2 * np.pi * k * freq * time)  # freq = 1/4 (period=4 years)
            #harmonics = np.vstack((sin_dta, cos_dta)).T
            dta_array[:, 2 * (k - 1)] = sin_dta
            dta_array[:, 2 * k - 1] = cos_dta
        sin_str = ['Sin'] * n_k
        cos_str = ['Cos'] * n_k
        factor_nums = np.linspace(1, n_k, n_k)
        sin_columns = ["%s%02d" % t for t in zip(sin_str, factor_nums)]
        cos_columns = ["%s%02d" % t for t in zip(cos_str, factor_nums)]
        iters = [iter(sin_columns), iter(cos_columns)]
        harmonics_cols =  list(it.next() for it in itertools.cycle(iters))
        dta_df = pd.DataFrame(index=x_data.index, data=dta_array, columns=harmonics_cols)
        return dta_df

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