import numpy as np
from skimage import metrics
from sklearn.metrics import precision_score, recall_score

from rampwf.score_types import BaseScoreType


def check_mask(mask):
    ''' assert that the given mask consists only of 0s and 1s '''
    assert np.all(np.isin(mask, [0, 1])), ('Cannot compute the score.'
                                           'Found values other than 0s and 1s')


# define the scores
class DiceCoeff(BaseScoreType):
    # Dice’s coefficient (DC), which describes the volume overlap between two
    # segmentations and is sensitive to the lesion size;
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='dice coeff', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_mask, y_pred_mask):
        check_mask(y_true_mask)
        check_mask(y_pred_mask)
        score = self._dice_coeff(y_true_mask, y_pred_mask)
        return score

    def _dice_coeff(self, y_true_mask, y_pred_mask):
        if (not np.any(y_pred_mask)) & (not np.any(y_true_mask)):
            # if there is no true mask in the truth and prediction
            return 1
        else:
            dice = (
                np.sum(np.logical_and(y_pred_mask, y_true_mask) * 2.0) /
                (np.sum(y_pred_mask) + np.sum(y_true_mask))
                )
        return dice


class Precision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='precision', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_mask, y_pred_mask):
        check_mask(y_true_mask)
        check_mask(y_pred_mask)
        if np.sum(y_pred_mask) == 0 and not np.sum(y_true_mask) == 0:
            return 0.0
        score = precision_score(y_true_mask.ravel(), y_pred_mask.ravel())
        return score


class Recall(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='recall', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_mask, y_pred_mask):
        check_mask(y_true_mask)
        check_mask(y_pred_mask)
        score = recall_score(y_true_mask.ravel(), y_pred_mask.ravel())
        return score


class HausdorffDistance(BaseScoreType):
    # recommened to use 95% percentile Hausdorff Distance which tolerates small
    # otliers
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='Hausdorff', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_mask, y_pred_mask):
        check_mask(y_true_mask)
        check_mask(y_pred_mask)
        score = metrics.hausdorff_distance(y_true_mask, y_pred_mask)
        return score


class AbsoluteVolumeDifference(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='AVD', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_mask, y_pred_mask):
        check_mask(y_true_mask)
        check_mask(y_pred_mask)
        score = np.abs(np.mean(y_true_mask) - np.mean(y_pred_mask))

        return score
