import glob
import os
import numpy as np
import pandas as pd
import rampwf as rw
from nilearn.image import load_img
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Iris classification'
_target_column_name = 'species'
_prediction_label_names = ['setosa', 'versicolor', 'virginica']
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.ClassificationError(name='error'),
    rw.score_types.NegativeLogLikelihood(name='nll'),
    rw.score_types.F1Above(name='f1_70', threshold=0.7),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_ids(path, split='train'):
    # FIXME: This should be replaced by hardcoding the IDs in train and test
    # CSV files.
    path_metadata = os.path.join(
        path, 'data', 'images', 'ATLAS_Meta-Data_Release_1.1_standard_mni.csv'
    )
    df = pd.read_csv(path_metadata)
    # FIXME: remove the duplicate for the moment
    ids = df['INDI Subject ID'].unique()
    rng = np.random.RandomState(42)
    train_id = rng.choice(ids, size=int(ids.size * 0.8), replace=False)
    test_id = np.setdiff1d(ids, train_id, assume_unique=True)
    if split == 'train':
        return train_id
    return test_id


def _read_brain_image(path, subject_id):
    path_metadata = os.path.join(
        path, 'data', 'images', 'ATLAS_Meta-Data_Release_1.1_standard_mni.csv'
    )
    df = pd.read_csv(path_metadata)
    # FIXME: just to take the first occurrence
    site_dir = df[df['INDI Subject ID'] == subject_id]['INDI Site ID'].iloc[0]
    subject_id = '{:06d}'.format(subject_id)
    path_brain_image = os.path.join(
        path, 'data', 'images', site_dir, subject_id, 't01',
        subject_id + '_t1w_deface_stx.nii.gz'
    )
    return load_img(path_brain_image).get_data()


def _read_stroke_segmentration(path, id):
    pass


def _read_data(path, split_ids):
    X = np.hstack([_read_brain_image(path, subject_id)
                   for subject_id in split_ids])
    return X


def get_train_data(path='.'):
    # generate the training IDs
    train_id = _read_ids(path, split='train')
    return _read_data(path, f_name)


def get_test_data(path='.'):
    # generate the training IDs
    test_id = _read_ids(path, split='test')
    return _read_data(path, f_name)
