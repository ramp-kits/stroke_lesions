from nilearn.image import load_img
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from skimage import filters


class FeatureExtractor(BaseEstimator, TransformerMixin):

    def _get_average(self, X):
        for idx, x_path in enumerate(X):
            x_data = load_img(x_path).get_data()

            if idx == 0:
                x_avg = x_data
            else:
                x_avg += x_data
        x_avg = x_avg / (idx + 1)  # make average
        return x_avg

    def _singlescale_basic_features(self, X, sigma, intensity=True,
                                    edges=True,
                                    avg_subtract=True):
        """ Features for a single value of the Gaussian blurring parameter
            ``sigma``
            simplified from code written by:
                Nicholas Esterer and Emmanuelle Gouillart
        """
        for idx, x_path in enumerate(X):
            features = []
            x_path = X[0]
            img = load_img(x_path).get_data()
            img_blur = filters.gaussian(img, sigma)

            if intensity:
                img_blur_reshaped = img_blur.reshape((1, -1))
                features.append(img_blur_reshaped)
            if edges:
                features.append(filters.sobel(img_blur).reshape(1, -1))
            #if avg_subtract:
            #    features.append((img - self._x_avg).reshape(1, -1))
            features = np.array(features)
            features = features.reshape((2,-1))

            if not idx:
                features_x = features
            else:
                features_x = np.hstack([features_x, features])
        return features_x

    def fit(self, X, y):
        self._x_avg = self._get_average(X)
        return self

    def transform(self, X):
        features = self._singlescale_basic_features(X=X, sigma=0.2)
        return features

from sklearn.linear_model import LogisticRegression
class PointEstimator(SGDClassifier):  #, ClassifierMixin, TransformerMixin):

    def fit(self, X, y):
        self.img_shape = y.shape[1:]
        y = y.reshape((1, -1))

        # self.clf = SGDClassifier(random_state=42) # output okish
        self.clf = LogisticRegression(random_state=42) # output 0
        # self.clf = 
        self.clf.fit(X.T, np.ravel(y))
        return self

    def predict_proba(self, X):

        y_pred = self.clf.predict(X.T)
        y = y_pred.reshape(-1, self.img_shape[0],
                           self.img_shape[1],
                           self.img_shape[2])
        print(np.sum(y))
        print(np.unique(y))
        return y


def get_estimator():
    # sets all the masks to all 1s

    extractor = FeatureExtractor()
    point_estimator = PointEstimator()

    pipeline = Pipeline([
        ('extractor', extractor),
        ('estimator', point_estimator)
    ])

    return pipeline
