import functools
import os
from nilearn.image import load_img
import numpy as np
import rampwf as rw
from sklearn.model_selection import ShuffleSplit
import warnings

from rampwf.prediction_types.base import BasePrediction
from rampwf.score_types import BaseScoreType

DATA_HOME = 'data'
RANDOM_STATE = 42

# Author: Maria Telenczuk <https://github.com/maikia>
# License: BSD 3 clause


class _MultiClass3d(BasePrediction):
    # y_pred should be 3 dimensional (x_len x y_len x z_len)
    def __init__(self, x_len, y_len, z_len, label_names,
                 y_pred=None, y_true=None, n_samples=None):
        # accepts only the predictions of classes 0 and 1
        self.x_len = x_len
        self.y_len = y_len
        self.z_len = z_len
        self.label_names = label_names

        if y_pred is not None:
            self.y_pred = np.array(y_pred)
        elif y_true is not None:
            self.y_pred = np.array(y_true)
        elif n_samples is not None:
            self.y_pred = np.empty((n_samples,
                                    self.x_len,
                                    self.y_len,
                                    self.z_len), dtype=float)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError(
                'Missing init argument: y_pred, y_true, or n_samples')
        self.check_y_pred_dimensions()

    def check_y_pred_dimensions(self):
        if len(self.y_pred.shape) != 4:
            raise ValueError(
                'Wrong y_pred dimensions: y_pred should be 4D, of size:'
                f'({self.n_samples} x {self.x_len} x {self.y_len}'
                f' x {self.z_len})'
                f'instead its shape is {self.y_pred.shape}')
        if self.y_pred.shape[1:] != (self.x_len, self.y_len, self.z_len):
            raise ValueError(
                'Wrong y_pred dimensions: y_pred should be'
                f' {self.x_len} x {self.y_len} x {self.z_len}'
                f' instead its shape is {self.y_pred.shape}')

    @classmethod
    def combine(cls, predictions_list, index_list=None):
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
            combined_predictions.y_pred[
                combined_predictions.y_pred < 0.5] = 0.0
            combined_predictions.y_pred[
                combined_predictions.y_pred >= 0.5] = 1.0

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


# define the scores
class DiceCoeff(BaseScoreType):
    # Dice’s coefficient (DC), which describes the volume overlap between two
    # segmentations and is sensitive to the lesion size;
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='dice coeff', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_mask, y_pred_mask):
        score = self._dice_coeff(y_true_mask, y_pred_mask)
        return score

    def _dice_coeff(self, y_true_mask, y_pred_mask):

        if (np.sum(y_pred_mask) == 0) & (np.sum(y_true_mask) == 0):
            return 1
        else:
            dice = (np.sum(
                (y_pred_mask == 1) & (y_true_mask == 1)
                ) * 2.0) / (np.sum(y_pred_mask) + np.sum(y_true_mask))
        return dice


def _partial_multiclass3d(cls=_MultiClass3d, **kwds):
    # this class partially inititates _MultiClass3d with given
    # keywords
    class _PartialMultiClass3d(_MultiClass3d):
        __init__ = functools.partialmethod(cls.__init__, **kwds)
    return _PartialMultiClass3d


def make_3dmulticlass(x_len, y_len, z_len, label_names):
    return _partial_multiclass3d(x_len=x_len, y_len=y_len, z_len=z_len,
                                 label_names=label_names)


# TODO: other score ideas:
# def average_symmetric_surface_distance(y_pred, y_true):
    # ASSD: denotes the average surface distance between two segmentations
    # 1. define the average surface distance (ASD)
    # - get all the surface voxels
    # 2. average over both directions
# def hausdorff_distance(y_pred, y_true):
    # a measure of the maximum surface distance, hence especially sensitive to
    # outliers maximum of all surface distances

problem_title = 'Stroke Lesion Segmentation'
_prediction_label_names = [0, 1]
_x_len, _y_len, _z_len = 193, 229, 193
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = make_3dmulticlass(x_len=_x_len, y_len=_y_len, z_len=_z_len,
                                label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.Estimator()


# TODO: dice coefficient
# TODO: scoring on the time of calculations?
score_types = [
    # rw.score_types.Accuracy(name='acc'),  # sklearn accuracy_score:
    # In multilabel classification, this function computes subset accuracy:
    # the set of labels predicted for a sample must exactly match the
    # corresponding set of labels in y_true.
    # rw.score_types.
    DiceCoeff(),
]


# cross validation
def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=RANDOM_STATE)
    return cv.split(X, y)


def _read_data(path, dir_name):
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

    dir_data = os.path.join(path, dir_name)
    list_subj_dirs = os.listdir(dir_data)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        # use only 5 subjects, otherwise take all
        list_subj_dirs = list_subj_dirs[:5]

    n_samples = len(list_subj_dirs)
    # we will be loading only the directory paths
    X = np.empty(n_samples, dtype='<U128')
    # we will be loading all the lesions arrays in
    y = np.empty((n_samples, _x_len, _y_len, _z_len))

    for idx, next_subj in enumerate(list_subj_dirs):
        X[idx] = os.path.join(dir_data, next_subj, 'T1.nii.gz')
        y_path = os.path.join(dir_data, next_subj, 'truth.nii.gz')
        y[idx, :] = load_img(y_path).get_fdata()
        # make sure that all the elements of y are in _prediction_label_names
        assert np.all(np.in1d(y, np.array(_prediction_label_names)))
    return X, y


def get_train_data(path='.'):
    path = os.path.join(path, DATA_HOME)
    return _read_data(path, 'train')


def get_test_data(path="."):
    path = os.path.join(path, DATA_HOME)
    return _read_data(path, 'test')
