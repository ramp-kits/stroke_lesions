
from sklearn.compose import make_column_transformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin


N_JOBS = 1

class Dummy(BaseEstimator, ClassifierMixin, TransformerMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        # returns y filled with only 1s
        y = np.arraylike(X)
        y[:] = 1
        return y

    def predict()


def get_estimator():
    dummy = Dummy()

    return dummy