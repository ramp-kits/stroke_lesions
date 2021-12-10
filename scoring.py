from sklearn.metrics import f1_score
import numpy as np
from rampwf.score_types.base import BaseScoreType
import inspect
from bids_loader import BIDSLoader

class DiceCoeff():
    def __init__(self,
                 name='Sørensen–Dice Coefficient',
                 precision=3):
        '''
        Scoring class for RAMP workflows. When called, returns the Sørensen–Dice coefficient. Note that this
        implementation allows for continuous values in the prediction.
        Parameters
        ----------
        name : str
            Name of the score; used for creating column headers.
        precision : str
            Numerical precision.
        '''
        # RAMP-related convention
        self.name = name
        self.precision = precision
        self.is_lower_the_better = False
        self.minimum = 0
        self.maximum = 1
        return

    def __call__(self,
                 y_true: tuple,
                 y_pred: np.array):

        return self.score_function(Y_true=y_true,
                                   Y_pred=y_pred)

    def score_function(self,
                       Y_true: np.array,
                       Y_pred: np.array):
        '''
        Returns the Sørensen–Dice coefficient for the input images. If multiple samples are given, the mean score
        is returned.
        Parameters
        ----------
        true_bids : BIDSPrediction
            BIDSPrediction with true_bids.y_true set as an array.
        pred : np.array
            Array containing the predicted labels of an image.

        Returns
        -------
        dice_coefficient : float
            Sørensen–Dice coefficient.
        '''
        y_true = np.array(Y_true.y_true)
        if(len(Y_pred.y_pred) == 0):
            return 0
        estimator = Y_pred.y_pred[0].estimator

        fscore = 0
        # Load example to ensure that the size fits
        dat = estimator.predict(BIDSLoader.load_image_tuple(Y_pred.y_pred[0].pred))
        fscore_list = []
        if (not self.check_y_pred_dimensions(y_true[0,...], dat)):
            raise (ValueError(f'Shape mismatch between y_true {y_true.shape} and y_pred {y_pred.shape}'))

        for idx, prediction_object in enumerate(Y_pred.y_pred):
            # First sample is already loaded; let's not waste the loading.
            if(idx != 0):
                dat = BIDSLoader.load_image_tuple(prediction_object.pred)
            # Note: If you want to get the weighted mean, use self._calc_score_parts
            sd_score = self.calc_score(dat, y_true[idx, ...])
            fscore += sd_score

        # Return the mean score
        return fscore/(idx+1)

    @staticmethod
    def calc_score(array_0: np.array,
                    array_1: np.array):
        '''
        Performs the calculation to get the Sørensen–Dice coefficient.
        Parameters
        ----------
        array_0 : np.array
            First array to score.
        array_1 : np.array
            Second array to score.

        Returns
        -------
        float
            Sørensen–Dice coefficient
        '''
        # Reshape to use dot product
        # NOTE: This computation of the coefficient allows for continuous values in the prediction.
        overlap, sum0, sum1 = DiceCoeff._calc_score_parts(array_0, array_1)
        sorenson = overlap / (sum0 + sum1)
        return sorenson

    @staticmethod
    def _calc_score_parts(array_0: np.array,
                    array_1: np.array):
        '''
        Computes the three parts of the Sørensen–Dice coefficient: overlap and 2 positives
        Parameters
        ----------
        array_0
        array_1

        Returns
        -------
        tuple
            Tuple containing (overlap, sum(array_0), sum(array_1)
        '''
        array_0_reshape = np.reshape(array_0, (1, np.prod(array_0.shape)))
        array_1_reshape = np.reshape(array_1, (np.prod((array_1.shape)), 1))
        overlap = 2*array_0_reshape @ array_1_reshape
        return (overlap[0][0], np.sum(array_0), np.sum(array_1))



    @staticmethod
    def check_y_pred_dimensions(array_0: np.array,
                                array_1: np.array):
        '''
        Checks that the dimensions of the inputs are consistent.
        Parameters
        ----------
        array_0 : np.array
            First array to check.
        array_1 : np.array
            Second array to check

        Returns
        -------
        bool
        '''
        if(array_0.shape != array_1.shape):
            return False
        else:
            return True
