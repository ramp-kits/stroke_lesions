import glob
import functools
import os
import numpy as np
from nilearn.image import load_img
from joblib import Memory
import rampwf as rw
from skimage import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import ShuffleSplit
import warnings

from rampwf.score_types import BaseScoreType
from rampwf.prediction_types.base import BasePrediction


DATA_HOME = 'data'
RANDOM_STATE = 42

mem = Memory('.')


@mem.cache
def load_img_data(fname):
    return load_img(fname).get_fdata()

# Author: Maria Telenczuk <https://github.com/maikia>
# License: BSD 3 clause


# -------- define the scores --------
def check_mask(mask):
    ''' assert that the given mask consists only of 0s and 1s '''
    assert np.all(np.isin(mask, [0, 1])), ('Cannot compute the score.'
                                           'Found values other than 0 and 1')


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

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        # y_true are paths
        return self.__call__(ground_truths.y_pred,
                             predictions.y_pred,
                             valid_indexes)

    def __call__(self, y_true_mask, y_pred_mask, valid_indexes=None):
        # calculate dice on each image separately to save on loading too much
        # memory at once
        score = np.empty(len(y_true_mask))

        for idx in range(len(y_true_mask)):
            if valid_indexes is None:
                valid_idx = slice(None, None)
            else:
                valid_idx = valid_indexes[idx]
            y_true = load_img_data(y_true_mask[idx])[valid_idx].astype('int32')
            y_pred = y_pred_mask[idx][valid_idx] * 1

            self.check_y_pred_dimensions(y_true, y_pred)
            check_mask(y_true)
            check_mask(y_pred)
            score[idx] = self._dice_coeff(y_true, y_pred)
        return np.nanmean(score)

    def _dice_coeff(self, y_true_mask, y_pred_mask):
        if len(y_true_mask) == 0 and len(y_pred_mask) == 0:
            return None
        if (not np.any(y_pred_mask)) & (not np.any(y_true_mask)):
            # if there is no true mask in the truth and prediction
            return 1

        smooth = 1.
        y_true_f = y_true_mask.flatten()
        y_pred_f = y_pred_mask.flatten()

        intersection = np.sum(y_true_f * y_pred_f)

        dice = ((2. * intersection + smooth) / (np.sum(y_true_f) +
                np.sum(y_pred_f) + smooth))
        return dice


class Precision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='precision', precision=3):
        self.name = name
        self.precision = precision

    def score_function(self, ground_truths, predictions, valid_indices=None):
        # y_true are paths
        return self.__call__(ground_truths.y_pred,
                             predictions.y_pred,
                             valid_indices=valid_indices)

    def __call__(self, y_true_mask, y_pred_mask, valid_indices):
        score = _calculate_score(y_true_mask, y_pred_mask,
                                 valid_indexes=valid_indices,
                                 func=self._calc_precision)
        return np.nanmean(score)

    def _calc_precision(self, y_true, y_pred):
        if np.sum(y_pred) == 0 and not np.sum(y_true) == 0:
            return 0.0
        score = precision_score(y_true.ravel(), y_pred.ravel())
        return score


class Recall(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='recall', precision=3):
        self.name = name
        self.precision = precision

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        # y_true are paths
        return self.__call__(ground_truths.y_pred,
                             predictions.y_pred,
                             valid_indexes)

    def __call__(self, y_true_mask, y_pred_mask, valid_indices=None):
        score = _calculate_score(y_true_mask, y_pred_mask,
                                 valid_indexes=valid_indices,
                                 func=self._calc_recall)
        return np.nanmean(score)

    def _calc_recall(self, y_true, y_pred):
        score = recall_score(y_true.ravel(), y_pred.ravel())
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

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        # y_true are paths
        return self.__call__(ground_truths.y_pred,
                             predictions.y_pred,
                             valid_indexes)

    def __call__(self, y_true_mask, y_pred_mask, valid_indices=None):
        score = _calculate_score(y_true_mask, y_pred_mask,
                                 valid_indexes=valid_indices,
                                 func=self._calc_hausdorff)
        return np.nanmean(score)

    def _calc_hausdorff(self, y_true, y_pred):
        score = metrics.hausdorff_distance(y_true, y_pred)
        return score


def _calculate_score(y_true_mask, y_pred_mask, func, valid_indexes=None):
    score = np.empty(len(y_true_mask))

    for idx in range(len(y_true_mask)):
        if valid_indexes is None:
            valid_idx = slice(None, None)
        else:
            valid_idx = valid_indexes[idx]
        y_true = load_img_data(y_true_mask[idx])[valid_idx].astype('int32')
        y_pred = y_pred_mask[idx][valid_idx] * 1

        # self.check_y_pred_dimensions(y_true, y_pred)
        check_mask(y_true)
        check_mask(y_pred)
        score[idx] = func(y_true, y_pred)

    return score


class AbsoluteVolumeDifference(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='AVD', precision=3):
        self.name = name
        self.precision = precision

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        # y_true are paths
        return self.__call__(ground_truths.y_pred,
                             predictions.y_pred,
                             valid_indexes)

    def __call__(self, y_true_mask, y_pred_mask, valid_indices=None):
        score = _calculate_score(y_true_mask, y_pred_mask,
                                 valid_indexes=valid_indices,
                                 func=self._calc_AVD)
        return np.nanmean(score)

    def _calc_AVD(self, y_true, y_pred):
        return np.abs(np.mean(y_true) - np.mean(y_pred))
# -------- end of define the scores --------


class _MultiClass3d(BasePrediction):
    # y_pred should be 3 dimensional (x_len x y_len x z_len)
    # y_true should be an array of paths
    def __init__(self, x_len, y_len, z_len, label_names,
                 y_pred=None, y_true=None, n_samples=None):
        # accepts only the predictions of classes 0 and 1
        self.x_len = x_len
        self.y_len = y_len
        self.z_len = z_len
        self.label_names = label_names
        self.n_samples = n_samples
        if y_pred is not None:
            self.y_pred = np.array(y_pred)
        elif y_true is not None:
            self.y_pred = np.array(y_true)
        elif self.n_samples is not None:
            self.y_pred = np.empty((self.n_samples,
                                    self.x_len,
                                    self.y_len,
                                    self.z_len), dtype=float)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError(
                'Missing init argument: y_pred, y_true, or n_samples')
        self.check_y_pred_dimensions()

    def check_y_pred_dimensions(self):
        # it should be an array with paths or a boolean array
        if len(self.y_pred.shape) not in [1, 4]:
            raise ValueError(
                'Wrong y_pred dimensions: y_pred should be 4D, of size:'
                f'({self.n_samples} x {self.x_len} x {self.y_len}'
                f' x {self.z_len})'
                f'instead its shape is {self.y_pred.shape}')
        if len(self.y_pred.shape) == 4 and\
           self.y_pred.shape[1:] != (self.x_len, self.y_len, self.z_len):
            raise ValueError(
                'Wrong y_pred dimensions: y_pred should be'
                f' {self.x_len} x {self.y_len} x {self.z_len}'
                f' instead its shape is {self.y_pred.shape}')

    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """For the sake of memory we want to operate on boolean masks and
        therefore estimators should already pass the boolean predictions.
        If this is not the case the threshold will be applied and the mask will
        be converted to bolean."""
        """Inherits from the base class where the scores are averaged.
        Here, averaged predictions < 0.5 will be set to 0.0 and averaged
        predictions >= 0.5 will be set to 1.0 so that `y_pred` will consist
        only of 0.0s and 1.0s.
        """
        # call the combine from the BasePrediction
        combined_predictions = super(
            _MultiClass3d, cls
            ).combine(
                predictions_list=predictions_list,
                index_list=index_list
                )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)

        return combined_predictions

    @property
    def valid_indexes(self):
        """Return valid indices (e.g., a cross-validation slice)."""
        if len(self.y_pred.shape) == 4:
            return ~np.isnan(self.y_pred)
        else:
            raise ValueError('y_pred.shape != 4 is not implemented')

    @property
    def _y_pred_label(self):
        return self.label_names[self.y_pred_label_index]


def _partial_multiclass3d(cls=_MultiClass3d, **kwds):
    # this class partially inititates _MultiClass3d with given
    # keywords
    class _PartialMultiClass3d(_MultiClass3d):
        __init__ = functools.partialmethod(cls.__init__, **kwds)
    return _PartialMultiClass3d


def make_3dmulticlass(x_len, y_len, z_len, label_names):
    return _partial_multiclass3d(x_len=x_len, y_len=y_len, z_len=z_len,
                                 label_names=label_names)


problem_title = 'Stroke Lesion Segmentation'
_prediction_label_names = [True, False]
_x_len, _y_len, _z_len = 197, 233, 189
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = make_3dmulticlass(x_len=_x_len, y_len=_y_len, z_len=_z_len,
                                label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    DiceCoeff(),
    AbsoluteVolumeDifference(),
    # HausdorffDistance(),
    Recall(),
    Precision()
]


# cross validation
def get_cv(X, y):
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        n_splits = 1
    else:
        n_splits = 8
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2,
                      random_state=RANDOM_STATE)
    return cv.split(X, y)


def _read_data(path):
    """
    Read and process data and labels.
    Parameters
    ----------
    path : path to directory that has 'data' subdir
    typ : {'train', 'test'}
    Returns
    -------
    X, y data
    """
    t1_name = '*T1.nii.gz'
    lesion_name = '_lesion.nii.gz'
    t1_names = glob.glob(os.path.join(path, t1_name))

    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        # use only 5 subjects, otherwise take all
        t1_names = t1_names[:3]
    X, y = [], []
    for idx, t1_next in enumerate(t1_names):
        X.append(t1_next)
        y_path = t1_next[:-(len(t1_name))] + lesion_name
        y.append(y_path)
    return np.array(X), np.array(y)


def get_train_data(path='.'):
    path = os.path.join(path, DATA_HOME)
    return _read_data(os.path.join(path, 'train'))


def get_test_data(path="."):
    path = os.path.join(path, DATA_HOME)
    return _read_data(os.path.join(path, 'test'))
