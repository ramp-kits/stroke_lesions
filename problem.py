import functools
import glob
import os
import numpy as np
import pandas as pd
import rampwf as rw
import scipy.sparse as sps
import warnings
from nilearn.image import load_img
from sklearn.model_selection import ShuffleSplit
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
            import pdb; pdb.set_trace()
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

    '''
    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """Inherits from the base class where the scores are averaged.
        Here, averaged predictions < 0.5 will be set to 0.0 and averaged
        predictions >= 0.5 will bexx = [(np.array([0, 1]), np.array([2])),
          (np.array([1, 2]), np.array([0]))]
    print(xx)set to 1.0 so that `y_pred` will consist
        only of 0.0s and 1.0s.
        """
        # call the combine from the BasePrediction
        combined_predictions = super(
            _MultiOutputClassification, cls
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
    '''


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
        if (np.sum(y_pred_mask)==0) & (np.sum(y_true_mask) == 0):
            return 1
        else:
            dice = (np.sum(
                (y_pred_mask==1) & (y_true_mask==1)
                )*2.0) / (np.sum(y_pred_mask) + np.sum(y_true_mask))
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
#_target_column_name = 'species'
_prediction_label_names = [0, 1]
_x_len, _y_len, _z_len = 193, 229, 193
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = make_3dmulticlass(x_len=_x_len, y_len=_y_len, z_len=_z_len,
                                label_names=_prediction_label_names)
# label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.Classifier()  # make_multiclass()  # rw.workflows.Classifier()


# TODO: dice coefficient
# TODO: scoring on the time of calculations?
score_types = [
    # rw.score_types.Accuracy(name='acc'),  # sklearn accuracy_score:
# In multilabel classification, this function computes subset accuracy:
# the set of labels predicted for a sample must exactly match the corresponding
# set of labels in y_true.
    # rw.score_types.
    DiceCoeff(),
]

# cross validation
def get_cv(X, y):
    # TODO: correct
    cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=RANDOM_STATE)
    '''
    X_size = sum(1 for _ in X)
    # import pdb; pdb.set_trace()
    xx = [(np.array([0, 1]), np.array([2])),
          (np.array([1, 2]), np.array([0]))]
    import pdb; pdb.set_trace()
    '''
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
    data_type = dir_name
    list_subj_dirs= os.listdir(dir_data)[:5]
    n_samples = len(list_subj_dirs)
    X = np.empty(n_samples, dtype='<U128')
    y = np.empty((n_samples, _x_len, _y_len, _z_len))
    for idx, next_subj in enumerate(list_subj_dirs):
        X[idx] = os.path.join(dir_data, next_subj, 'T1.nii.gz')
        y_path = os.path.join(dir_data, next_subj, 'truth.nii.gz')
        y[idx, :] = load_img(y_path).get_data()

    return X, y

def get_train_data(path='.'):
    path = os.path.join(path, DATA_HOME)
    return _read_data(path, 'train')

def get_test_data(path="data"):
    path = os.path.join(path, DATA_HOME)
    return _read_data(path, 'test')


# import submissions.starting_kit.keras_segmentation_classifier as classifier
'''
class SimplifiedSegmentationClassifier(object):
    """
    SimplifiedSegmentationClassifier workflow.
    This workflow is used to train image segmentation tasks, typically when
    the dataset cannot be stored in memory. It is altered version of SimplifiedImageClassifier
    Submissions need to contain one file, which by default by is named
    segmentation_classifier.py (it can be modified by changing
    `workflow_element_names`).
    image_classifier.py needs an `SegmentationClassifier` class, which implements
    `fit` and `predict`, where both `fit` and `predict` take
    as input an instance of `ImageLoader`.
    Parameters
    ==========
    n_classes : int
    data_file : file in data/images/ with all the patient ids and in which Site dir they can be found
        Total number of classes.
    """

    def __init__(self, n_classes=[0,1], data_file='ATLAS_Meta-Data_Release_1.1_standard_mni.csv', workflow_element_names=['segmentation_classifier']):
        self.n_classes = n_classes
        self.element_names = workflow_element_names
        self.data_file = data_file

    def train_submission(self, module_path, patient_ids):
        """Train an image classifier.
        module_path : str
            module where the submission is. the folder of the module
            have to contain segmentation_classifier.py. (e.g. starting_kit)
        patient_idxs : ArrayContainer vector of int
             patient ids (as in the data/images/<self.data_file> file)
             which are to be used for training
        """

        # FIXME: when added to workflow add those lines:
        #segmentation_classifier = import_file(module_path, self.element_names[0])
        #clf = segmentation_classifier.SegmentationClassifier()
        clf = classifier.KerasSegmentationClassifier()
        # load image one by one
        #for patient_id in patient_idxs:
        #    X = self._read_brain_image(module_path, patient_id)
        #    y_true = self._read_stroke_segmentation(module_path, patient_id)
        #    fitit = clf.fit(X, y_true)
        #    del X
        #    del y_true
        #return clf

        img_loader = ImageLoader(
            patient_ids,
            #X_array[train_is], y_array[train_is],
            #folder=folder,
            n_classes=self.n_classes
        )
        clf.fit(img_loader)
        return clf

    def test_submission(self, module_path, trained_model, patient_idxs):
        """Test an image classifier.
        trained_model : tuple (function, Classifier)
            tuple of a trained model returned by `train_submission`.
        patient_idxs : ArrayContainer of int
            vector of image IDs to test on.
            (it is named X_array to be coherent with the current API,
             but as said here, it does not represent the data itself,
             only image IDs).
        """
        # get images one by one
        clf = trained_model

        dices = np.zeros(len(patient_idxs))
        # load image one by one
        for idx, patient_id in enumerate(patient_idxs):
            X = self._read_brain_image(module_path, patient_id)
            y_true = self._read_stroke_segmentation(module_path, patient_id)
            y_pred = clf.predict(X)
            dices[idx] = dice_coefficient(y_pred, y_true)
            del X
            del y_true
            del y_pred
        return dices
'''
'''
import os
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose
from keras.layers import Concatenate
from keras import Model

def fit_simple():

    inputs = Input((197, 233, 189, 1))
    x = BatchNormalization()(inputs)
    # downsampling
    down1conv1 = Conv3D(2, (3, 3, 3), activation='relu', padding='same')(x)
    down1conv1 = Conv3D(2, (3, 3, 3), activation='relu', padding='same')(down1conv1)
    down1pool = MaxPooling3D((2, 2, 2))(down1conv1)
    #middle
    mid_conv1 = Conv3D(2, (3, 3, 3), activation='relu', padding='same')(down1pool)
    mid_conv1 = Conv3D(2, (3, 3, 3), activation='relu', padding='same')(mid_conv1)

    # upsampling
    up1deconv = Conv3DTranspose(2, (3, 3, 3), strides=(2,2,2), activation='relu')(mid_conv1)
    up1concat = Concatenate()([up1deconv, down1conv1])
    up1conv1 = Conv3D(2, (3,3,3), activation='relu', padding='same')(up1concat)
    up1conv1 = Conv3D(2, (3,3,3), activation='relu', padding='same')(up1conv1)
    output = Conv3D(1, (3,3,3), activation='softmax', padding='same')(up1conv1)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='rmsprop',
                 loss='mean_squared_error',
                 metrics=['accuracy'])

    train_suffix='_LesionSmooth_*.nii.gz'
    train_id = get_train_data(path='.')
    brain_image = _read_brain_image('.', train_id[1])
    mask = _read_stroke_segmentation('.', train_id[1])

    #model.fit(brain_image[None, ..., None], mask[None, ..., None].astype(bool))
    #model.fit_on_batch
    #return model
    # train the network
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)
'''