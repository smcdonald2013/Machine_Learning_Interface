import unittest
import matplotlib
matplotlib.use('AGG')
import Machine_Learning_Interface.ols_regression as os
import Machine_Learning_Interface.regularized_regression as rr
import Machine_Learning_Interface.svm_regression as sm
import Machine_Learning_Interface.decision_tree_regression as dt
import Machine_Learning_Interface.kalman_regression as kr
import Machine_Learning_Interface.logistic_regression as lr
import Machine_Learning_Interface.lda_classification as lda
import Machine_Learning_Interface.qda_classification as qda
import Machine_Learning_Interface.naive_bayes_classification as nb
import Machine_Learning_Interface.svm_classification as svc
import pandas as pd
from pandas.util import testing
import numpy as np
from sklearn import datasets
state = np.random.seed(10)

class CustomTestCase(unittest.TestCase):
    def assertSeriesEqual(self, model_coefs, true_coefs):
        testing.assert_series_equal(model_coefs, true_coefs, check_less_precise=True)

class RegressionTests(CustomTestCase):
    def setUp(self):
        n_samples =  5
        self.x_data = np.array([list(range(1,n_samples+1))]).reshape(n_samples,1)
        feat2 = self.x_data * 3
        self.x_multi = np.array([self.x_data, feat2]).reshape(n_samples, 2)
        self.y_data = (self.x_data*2).reshape(n_samples)

    def test_ols(self):
        model = os.OLSRegression(intercept=True, scale=False)

        #Univariate regression
        model, predictions = model_estimating(model, self.x_data, self.y_data)
        true_coefs = pd.Series(data=[2.0, 0.0], index=["0", "intercept"], name = 'coefficients')
        self.assertSeriesEqual(model.coefs, true_coefs)
        true_pred = pd.Series(data=[2.0, 4.0, 6.0, 8.0, 10.0], name = 'predictions', dtype='float64')
        self.assertSeriesEqual(predictions, true_pred)

        #Multivariate regression
        model, predictions = model_estimating(model, self.x_multi, self.y_data)
        true_coefs = pd.Series(data=[0.655352, 0.054830, 2.099217], index=["0", "1", "intercept"], name = 'coefficients')
        self.assertSeriesEqual(model.coefs, true_coefs)
        true_pred = pd.Series(data=[2.86422976501, 4.28459530026, 5.54046997389, 6.52480417755, 10.7859007833], name = 'predictions', dtype='float64')
        self.assertSeriesEqual(predictions, true_pred)

    def test_lasso(self):
        model = rr.LassoRegression(intercept=True, cv_folds=5, scale=False, solver='Coordinate Descent', selection='cyclic' ,n_jobs=-1, n_alphas=100)

        #Univariate regression
        model, predictions = model_estimating(model, self.x_data, self.y_data)
        true_coefs = pd.Series(data=[1.998, 0.006], index=["0", "intercept"], name = 'coefficients')
        self.assertSeriesEqual(model.coefs, true_coefs)
        true_pred = pd.Series(data=[2.004, 4.002, 6.000, 7.998, 9.996], name = 'predictions', dtype='float64')
        self.assertSeriesEqual(predictions, true_pred)

        #Multivariate regression
        model, predictions = model_estimating(model, self.x_multi, self.y_data)
        true_coefs = pd.Series(data=[0.299420, 00.226150, 2.890545], index=["0", "1", "intercept"], name = 'coefficients')
        self.assertSeriesEqual(model.coefs, true_coefs)
        true_pred = pd.Series(data=[3.64226433794, 4.69340329472, 5.06609289428, 6.72241130136, 9.8758281717], name = 'predictions', dtype='float64')
        self.assertSeriesEqual(predictions, true_pred)

    def test_ridge(self):
        model = rr.RidgeRegression(intercept=True, cv_folds=5, scale=False)

        #Univariate regression
        model, predictions = model_estimating(model, self.x_data, self.y_data)
        true_coefs = pd.Series(data=[2.0, 0.0], index=["0", "intercept"], name = 'coefficients')
        self.assertSeriesEqual(model.coefs, true_coefs)
        true_pred = pd.Series(data=[2.00000000004, 4.00000000002, 6.0, 7.99999999998, 9.99999999996], name = 'predictions', dtype='float64')
        self.assertSeriesEqual(predictions, true_pred)

        #Multivariate regression
        model, predictions = model_estimating(model, self.x_multi, self.y_data)
        true_coefs = pd.Series(data=[0.65535248041, 0.0548302872119, 2.09921671019], index=["0", "1", "intercept"], name = 'coefficients')
        self.assertSeriesEqual(model.coefs, true_coefs)
        true_pred = pd.Series(data=[2.86422976502, 4.28459530027, 5.54046997387, 6.524804177556, 10.7859007833], name = 'predictions', dtype='float64')
        self.assertSeriesEqual(predictions, true_pred)

    def test_elastic_net(self):
        model = rr.ElasticNetRegression(intercept=True, cv_folds=5, scale=True, l1_ratio=[.1, .5, .8, .9, .95])

        #Univariate regression
        model, predictions = model_estimating(model, self.x_data, self.y_data)
        true_coefs = pd.Series(data=[2.825178, 6.0], index=["0", "intercept"], name = 'coefficients')
        self.assertSeriesEqual(model.coefs, true_coefs)
        true_pred = pd.Series(data=[2.004, 4.002, 6.000, 7.998, 9.996], name = 'predictions', dtype='float64')
        self.assertSeriesEqual(predictions, true_pred)

        #Multivariate regression
        model, predictions = model_estimating(model, self.x_multi, self.y_data)
        true_coefs = pd.Series(data=[1.29025770301, 0.974112495429, 6.0], index=["0", "1", "intercept"], name = 'coefficients')
        self.assertSeriesEqual(model.coefs, true_coefs)
        true_pred = pd.Series(data=[3.54845198097, 4.64450098222, 5.13694673924, 6.69097664691, 9.97912365066], name = 'predictions', dtype='float64')
        self.assertSeriesEqual(predictions, true_pred)

    def test_svr(self):
        model = sm.SVR(scale=True, kernel='rbf', parameters=[{'C' : np.logspace(-3, 3, 7), 'epsilon' : np.logspace(-3, 3, 7)}], cv_folds=3)

        #Univariate regression
        model, predictions = model_estimating(model, self.x_data, self.y_data)
        true_pred = pd.Series(data=[2.00099997959, 3.99870905512, 6.00129094488, 7.99900002041, 7.25513636298], name = 'predictions')
        self.assertSeriesEqual(predictions, true_pred)

        #Multivariate regression
        model, predictions = model_estimating(model, self.x_multi, self.y_data)
        true_pred = pd.Series(data=[2.00134958652, 4.00100147809, 5.99900137882, 7.99864742333, 5.75672884513], name = 'predictions', dtype='float64')
        self.assertSeriesEqual(predictions, true_pred)

    def test_dtr(self):
        model = dt.DTR(reg_type = 'rand_forest', n_estimators=10, cv_folds=5, parameters=[{'max_depth' : np.linspace(1, 15, 15)}], random_state=10)

        #Univariate regression
        model, predictions = model_estimating(model, self.x_data, self.y_data)
        true_pred = pd.Series(data=[2.4, 3.6, 5.4, 7.0, 8.6], name = 'predictions')
        self.assertSeriesEqual(predictions, true_pred)

        #Multivariate regression
        model, predictions = model_estimating(model, self.x_multi, self.y_data)
        true_pred = pd.Series(data=[2.7, 4.0, 5.3, 6.3, 8.6], name = 'predictions', dtype='float64')
        self.assertSeriesEqual(predictions, true_pred)


    def test_kalman(self):
        em    = ['transition_covariance', 'observation_covariance', 'initial_state_mean', 'initial_state_covariance']
        model = kr.KalmanRegression(intercept=True, scale=False, em_vars=em)
        model, predictions = model_estimating(model, self.x_data, self.y_data)
        true_pred = pd.Series(data=[2.007046, 4.005116, 6.003186, 8.001256, 9.999327], name = 'predictions')
        self.assertSeriesEqual(predictions, true_pred)

class ClassificationTests(CustomTestCase):
    def setUp(self):
        feat1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(10, 1)
        feat2 = feat1 * 2
        self.x_data = np.array([feat1, feat2]).reshape(10, 2)
        self.y_data = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1]).reshape(10)
        self.y_multiclass = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3,3]).reshape(10)

    def test_logistic(self):
        model = lr.LogisticRegression(intercept=True, scale=False, cv_folds=3, penalized=True, prob=True)
        model, predictions = model_estimating(model, self.x_data, self.y_data)
        true_pred = pd.Series(data=[1, 1, 1, 1, 1, 1, 1, 1, -1, -1], name = 'predictions', dtype='int32')
        self.assertSeriesEqual(predictions, true_pred)

    def test_lda(self):
        model = lda.LDA(scale=False, intercept=False, prob=True)
        model, predictions = model_estimating(model, self.x_data, self.y_data)
        true_pred = pd.Series(data=[1, 1, 1, 1, 1, 1, -1, -1, -1, -1], name = 'predictions', dtype='int32')
        self.assertSeriesEqual(predictions, true_pred)

    def test_qda(self):
        model = qda.QDA(scale=True, intercept=False, prob=True)
        model, predictions = model_estimating(model, self.x_data, self.y_data)
        true_pred = pd.Series(data=[1, 1, 1, 1, 1, -1, -1, -1, -1, -1], name = 'predictions', dtype='int32')
        self.assertSeriesEqual(predictions, true_pred)

    def test_qda_multi(self):
        model = qda.QDA(scale=True, intercept=False, prob=True)
        model, predictions = model_estimating(model, self.x_data, self.y_multiclass)
        true_pred = pd.Series(data=[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], name = 'predictions', dtype='int32')
        self.assertSeriesEqual(predictions, true_pred)

    def test_nb(self):
        model = nb.NaiveBayesClassification(scale=True, intercept=False, prob=False)
        model, predictions = model_estimating(model, self.x_data, self.y_data)
        true_pred = pd.Series(data=[1, 1, 1, 1, -1, 1, 1, -1, -1, -1], name = 'predictions', dtype='int32')
        self.assertSeriesEqual(predictions, true_pred)

    def test_svc(self):
        #n_samples = 1000
        #feat1 = np.array(list(range(n_samples))).reshape(n_samples, 1)
        #feat2 = feat1 * 2
        #x_data = np.array([feat1, feat2]).reshape(n_samples, 2)
        #pos = [1]*(n_samples/2)
        #neg = [-1]*(n_samples/2)
        #y_data = np.array(pos+neg).reshape(n_samples)
        #self.y_multiclass = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3,3]).reshape(15)
        x_data, y_data = datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, n_classes=2, random_state=10)
        model = svc.SVC(scale=True, prob=True, kernel='linear', parameters=[{'C' : np.logspace(-3, 3, 7)}], cv_folds=3)
        model, predictions = model_estimating(model, x_data, y_data)
        print(predictions)
        #true_pred = pd.Series(data=[1, 1, 1, 1, -1, 1, 1, -1, -1, -1], name = 'predictions', dtype='int32')
        true_pred = pd.Series(data=y_data, name = 'predictions', dtype='int32')
        self.assertSeriesEqual(predictions, true_pred)

def model_estimating(model, x_data, y_data):
    model.fit(pd.DataFrame(x_data),pd.Series(y_data))
    model.diagnostics()
    predictions = model.predict(pd.DataFrame(x_data))
    return model, predictions

def suite():
    suite = unittest.TestSuite()
    #suite.addTest(ClassificationTests('test_svc'))
    #suite.addTest(ClassificationTests('test_nb'))
    suite.addTest(RegressionTests('test_ols'))
    suite.addTest(RegressionTests('test_lasso'))
    #suite.addTest(RegressionTests('test_ridge'))
    #suite.addTest(RegressionTests('test_elastic_net'))
    #suite.addTest(RegressionTests('test_svr'))
    #suite.addTest(RegressionTests('test_dtr'))
    return suite

if __name__ == '__main__':
    #unittest.main()
    runner = unittest.TextTestRunner()
    runner.run(suite())
