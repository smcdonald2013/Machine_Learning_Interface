import numpy as np
import pandas as pd 
from sklearn import grid_search, learning_curve, tree, ensemble
import matplotlib.pyplot as plt
from .base_models import Regression
from . import scikit_mixin

class DTR(Regression):
    """Class for Decision Tree Regression Models. 

    Parameters
    ----------
    intercept : boolean
        Whether to fit an intercept to the model.
    scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    kernel : string 
        Kernel used for SVR. Options from sklearn are 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed', or callable. 
    parameters : List
        Parameters to be used for cross-validation. Ignored is cv_folds is None. Must be in the grid search form used by sklearn, i.e. 
        parameters = [{'kernel': ['linear', 'rbf'], 'C': [.1, 1, 10], 'epsilon' : [.1, 1, 10]}]
    cv_folds : int        
        Number of folds for cross validation. If None, model is fit on entire dataset. 

    Attributes
    ----------
    self.intercept : boolean
        Whether to fit an intercept to the model. Ignored if model_provided is not None. 
    self.scale : boolean
        Whether to scale the data so each variable has mean=0 and variance=1
    self.cv_folds : int        
        Number of folds for cross validation. If None, 
    """
    def __init__(self, intercept=False, scale=False, cv_folds=None, parameters=None, reg_type='tree',**kwargs):
        self.intercept      = intercept
        self.scale          = scale
        self.reg_type       = reg_type
        self.cv_folds       = cv_folds
        self.parameters     = parameters
        self.kwargs         = kwargs 

    def _estimate_model(self): 
        if self.reg_type=='tree':
            self.underlying = tree.DecisionTreeRegressor(**self.kwargs)
        elif self.reg_type=='rand_forest':
            self.underlying = ensemble.RandomForestRegressor(**self.kwargs) #Should provide n_estimators as kwarg
        elif self.reg_type=='ex_rand_forest':
            self.underlying = ensemble.ExtraTreesRegressor(**self.kwargs) #Should provide n_estimators as kwarg
        else: 
            print('Invalid reg_type')
        if self.cv_folds is not None: 
            self.model = grid_search.GridSearchCV(self.underlying, self.parameters, cv=self.cv_folds, scoring='mean_squared_error')
        else:
            self.model = self.underlying
        self.model.fit(self.x_train, self.y_train)
        return self.model

    def _estimate_coefficients(self):
        if self.kernel=='linear':
            coef_array =  np.append(self.model.coef_,self.model.intercept_)
            coef_names = np.append(self.x_train.columns, 'intercept')
            coef_df = pd.Series(data=coef_array, index=coef_names, name = 'coefficients')
        else: 
            coef_df = None
        return coef_df

    def _estimate_fittedvalues(self):
        yhat = self.predict(self.x_train)
        return yhat

    def diagnostics(self):
        super(DTR, self).diagnostics() 
        if self.cv_folds is not None:
            self.cv_params = self.model.best_params_
            self.grid_scores = self.model.grid_scores_
            scikit_mixin.learning_curve_plot(estimator=self.underlying, title='Decision Tree Learning Curve', X=self.x_train, y=self.y_train, cv=5, scoring='mean_squared_error')
            scikit_mixin.validation_plot(estimator=self.underlying, title='Validation Plot', X=self.x_train, y=self.y_train, cv=5, scoring='mean_squared_error', param_name='max_depth', param_range=self.parameters[0]['max_depth'], cv_param=self.model.best_params_['max_depth'], scale='linear')
        if self.reg_type=='tree':
            self.tree_image()

    def predict(self, x_val):
        super(DTR, self).predict(x_val) 
        val_pred = self.model.predict(self.x_val)
        val_df   = pd.Series(index=self.x_val.index, data=val_pred, name='predictions')
        return val_df    

    def tree_image(self, output_option='inline'):
        if output_option=='inline':
            from IPython.display import Image, display 
            from sklearn.externals.six import StringIO
            import pydot
            dot_data = StringIO()  
            tree.export_graphviz(self.model, out_file=dot_data, filled=True, rounded=True, special_characters=True)  
            graph = pydot.graph_from_dot_data(dot_data.getvalue())  
            return display(Image(graph[0].create_png())) 
        elif output_option=='pdf':
             print('Design this later!')
        else:
            print('Unsupported output_option')
            