from nilearn.image import load_img
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


class Subtract(BaseEstimator, ClassifierMixin, TransformerMixin):

    def _get_average(self, X):
        for idx, x_path in enumerate(X):
            x_data = load_img(x_path).get_data()

            if idx == 0:
                x_avg = x_data
            else:
                x_avg += x_data
        x_avg = x_avg / (idx + 1)  # make average
        return x_avg


    def fit(self, X, y):
        self._x_avg = self._get_average(X)

        return self

    def transform(self, X, y):
        # subtract image(s) from average
        for idx, x_path in enumerate(X):
            x_data = load_img(x_path).get_data()
            import pdb; pdb.set_trace()
            x_data - self._x_avg
            # TODO: return subtraced Xs
        return X



class PointEstimator(BaseEstimator, ClassifierMixin, TransformerMixin):

    def fit(self, X, y):


    def predict_proba(self, X):
        # returns y filled with only 1s
        
        import pdb; pdb.set_trace()
        x_data = x_data/(idx + 1)
        x_shape = x_data.shape
        y = np.ones((len(X), x_shape[0], x_shape[1], x_shape[2]))

        return y


def get_estimator():

    # sets all the masks to all 1s

    subtr = Subtract()


    pipeline = Pipeline([
        ('classifier', subtr),
        # add multioutput
        ('linear_regression', LinearRegression())
    ])

    return pipeline
