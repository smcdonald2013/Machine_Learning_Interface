import theano
#theano.config.gcc.cxxflags='-march=core2'
#theano.config.gcc="C:\TDM-GCC-64\bin\\g++.exe"
import pymc3
import scipy
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import abc
from .base_models import Regression

class BayesianRegression(Regression):
    """Class for Bayesian Regression models. 
    Bayesian models are a wide field. This class implements
    a default linear Gaussian model, but also accepts user 
    supplied models. 

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    niter : int
        Number of iterations to use when fitting the model. 
    model_provided : pymc3 Model object
        Model to use if default linear Gaussian model is not desired. 

    Attributes
    ----------
    intercept : boolean
        Whether an intercept was fit to the model. 
    scale : boolean
        Whether the data was scaled before fitting. 
    niter : int
        Number of iterations used when fitting the model. 
    """
    def __init__(self, intercept=False, scale=False, niter=500, model_provided=None):
        self.intercept = intercept
        self.scale = scale
        self.niter = niter
        self.model_provided = model_provided

    def _estimate_model(self):
        #If user provides model, use that. Otherwise, create default Bayesian Model
        self.x_shared = theano.shared(self.x_train.values)
        if self.model_provided is not None:
            self.model = self.model_provided
        else:
            self.model = pymc3.Model()
            with self.model:

                # Priors for unknown model parameters
                alpha = pymc3.Normal('alpha', mu=0, sd=1)
                beta = pymc3.Normal('beta', mu=0, sd=1, shape=self.number_feat)
                sigma = pymc3.HalfNormal('sigma', sd=1)

                # Expected value of outcome
                #mu = alpha + x_shared[:,0]*beta[0] + x_shared[:,1]*beta[1] + x_shared[:,2]*beta[2] + x_shared[:,3]*beta[3] + x_shared[:,4]*beta[4] + x_shared[:,5]*beta[5]  + x_shared[:,6]*beta[6] + x_shared[:,7]*beta[7] 
                mu = alpha + theano.tensor.dot(self.x_shared, beta)

                # Likelihood (sampling distribution) of observations
                Y_obs = pymc3.Normal('Y_obs', mu=mu, sd=sigma, observed=self.y_train.values)

        with self.model:
            self.start = pymc3.find_MAP(fmin=scipy.optimize.fmin_powell)
            step = pymc3.NUTS(scaling=self.start)
            self.trace = pymc3.sample(self.niter, step)

        return self.model

    def diagnostics(self):
        super(BayesianRegression, self).diagnostics() 
        self.coefs = self._estimate_coefficients()
        trace = pymc3.traceplot(self.trace)
        self.map_estimate = pymc3.find_MAP(model=self.model)
        
    def _estimate_fittedvalues(self):
        self.x_shared.set_value(self.x_train.values)
        ppc = self._run_ppc(self.trace, model=self.model)
        fitted_values = pd.Series(self.x_train.dot(self.start['beta'])+self.start['alpha'], name='fitted')
        return fitted_values

    def _estimate_coefficients(self):
        coef_a =  np.append(self.start['beta'],self.start['alpha'])
        coef_names = np.append(self.x_train.columns, 'intercept')
        coef_df = pd.Series(data=coef_a, index=coef_names, name = 'coefficients')
        return coef_df

    def predict(self, x_val):
        super(BayesianRegression, self).predict(x_val) 
        self.x_shared.set_value(self.x_val.values)
        ppc = self._run_ppc(self.trace, model=self.model)
        val_pred = self.x_val.values.dot(self.start['beta'])+self.start['alpha']
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df

    def _run_ppc(self, trace, samples=100, model=None):
        """Generate Posterior Predictive samples from a model given a trace."""
        ppc = collections.defaultdict(list)
        for idx in np.random.randint(0, len(trace), samples):
            param = trace[idx]
            for obs in model.observed_RVs:
                ppc[obs.name].append(obs.distribution.random(point=param))
        return ppc   