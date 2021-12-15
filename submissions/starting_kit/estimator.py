import numpy as np
from sklearn.base import BaseEstimator


class BIDSEstimator(BaseEstimator):
    '''
    Estimator sample for RAMP stroke lesion segmentation.
    '''
    def __init__(self):
        '''
        Initialize estimator values (e.g. starting learning rate) here.
        '''
        return

    def fit(self,
            X: np.array,
            y: np.array):
        '''
        Fit the estimator using the input data (X) and target (y). Assumes that all data is present. Optional.
        This estimator in particular does nothing.
        Parameters
        ----------
        X : np.array
            Data of the form (n_samples, n_channels, *image.shape)
        y : np.array
            Target (labels) of the form (n_samples, n_channels, *image.shape)

        Returns
        -------
        None
        '''
        return

    def fit_partial(self, X, y):
        '''
        Fit the estimator using the input data (X) and target (y). Assumes that the inputs represent only a fraction
        of the data and that it will be called multiple times while using the dataset. I.e., learning rates and adaptive
        parameters should not be entirely recalculated with each call to this method. Required.
        This estimator in particular does nothing.
        Parameters
        ----------
        X : np.array
            Data of the form (n_samples, n_channels, *image.shape)
        y : np.array
            Target (labels) of the form (n_samples, n_channels, *image.shape)

        Returns
        -------
        None
        '''

        # Apply pre-processing to X
        # Feed to estimator
        return

    def predict_proba(self, X):
        '''
        Applies the data to the estimator to produce a prediction. The output can be continuous to represent the
        relative confidence the estimator has in the prediction. Optional.
        Typically, correct but uncertain predictions are rewarded less. Similarly, incorrect but uncertain predictions
        are punished less severely.
        This estimator always returns 1.
        Parameters
        ----------
        X : np.array
            Data of the form (n_samples, n_channels, *image.shape)

        Returns
        -------
        np.array
            Prediction made by the estimator.
        '''
        y = np.ones(X.shape, dtype=bool)
        return y

    def predict(self, X):
        '''
            Applies the data to the estimator to produce a prediction. The output type is expected to match the problem.
            I.e., classification problems should have categorical predictions. Required.
            This estimator always returns 1.
            Parameters
            ----------
            X : np.array
                Data of the form (n_samples, n_channels, *image.shape)

            Returns
            -------
            np.array
                Prediction made by the estimator.
            '''
        y = np.ones(X.shape, dtype=bool)
        return y

def get_estimator():
    return BIDSEstimator