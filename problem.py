import sklearn
import os
import numpy as np
import warnings

from stroke import prediction
from stroke.bids_workflow import BIDSWorkflow
from stroke.scoring import DiceCoeff
from stroke import stroke_config
from stroke.bids_loader import BIDSLoader

problem_title = "ATLAS Stroke Lesion Segmentation"

# Define workflow; this determines how data is trained + tested.
workflow = BIDSWorkflow()
Predictions = prediction.BIDSPrediction  # Class containing data + targets
# Scores to evaluate; object is instantiated because RAMP expects some
# fields to be defined
score_types = [DiceCoeff()]


def get_cv(X: np.array,
           y: np.array):
    '''
    Returns the train/test split for each fold of k-fold cross-validation.
    Parameters
    ----------
    X : np.array
        Array with the first dimension being the number of samples in the training set. Data is not used; a zero-array
        suffices.
    y : np.array
        Same as X.

    Returns
    -------
    list [list]
        List of train/test indices for each fold of k-fold cross-validation.
    '''
    strat = sklearn.model_selection.ShuffleSplit(
        n_splits=stroke_config.cross_validation['n_splits'],
        train_size=stroke_config.cross_validation['train_size'],
        random_state=stroke_config.cross_validation['random_state'])
    return strat.split(X, y)


def get_train_data(path: str):
    '''
    Returns the list of training data and the corresponding targets.
    Parameters
    ----------
    path : str

    Returns
    -------
    tuple (data_list, target_list)
    '''

    # BIDS parsing is slow, especially for larger sets. The config file is loaded once, but we don't have a way of
    # passing the command-line argument 'path' to it.
    # If 'path' is the same as in the config file, we only need to load it once
    # Otherwise; continue, but warn user and give instructions on how to
    # optimize settings.
    if(path == '.' or path == './'):
        path = 'data'
    if(os.path.abspath(path) == os.path.abspath(stroke_config.data_path)):
        return stroke_config.bids_loader_train.data_list, stroke_config.bids_loader_train.target_list
    else:
        warnings.warn(
            f'Data path differs from that in the config file; to reduce the amount of time spent loading '
            f'files, modify config.py: data_path = {path}')
        training_dir = os.path.join(path, stroke_config.training['dir_name'])
        bids_loader_train = BIDSLoader(root_dir=training_dir,
                                       data_entities=[{'subject': '',
                                                       'session': '',
                                                       'desc': 'T1FinalResampledNorm'}],
                                       target_entities=[{'label': 'L',
                                                         'desc': 'T1lesion',
                                                         'suffix': 'mask'}],
                                       data_derivatives_names=['ATLAS'],
                                       target_derivatives_names=['ATLAS'],
                                       label_names=['not lesion', 'lesion'],
                                       batch_size=stroke_config.training['batch_size'])

        if(stroke_config.is_quick_test):
            bids_loader_train.data_list[:stroke_config.num_subjects_quick_test], \
            bids_loader_train.target_list[:stroke_config.num_subjects_quick_test]
        else:
            return bids_loader_train.data_list, bids_loader_train.target_list


def get_test_data(path: str):
    '''
    Returns the list of testing data and the corresponding targets.
    Parameters
    ----------
    path : str

    Returns
    -------
    tuple (data_list, target_list)
    '''
    # BIDS parsing is slow, especially for larger sets. The config file is loaded once, but we don't have a way of
    # passing the command-line argument 'path' to it.
    # If 'path' is the same as in the config file, we only need to load it once
    # Otherwise; continue, but warn user and give instructions on how to
    # optimize settings.
    if (path == '.' or path == './'):
        path = 'data'
    if (os.path.abspath(path) == os.path.abspath(stroke_config.data_path)):
        return stroke_config.bids_loader_test.data_list, stroke_config.bids_loader_test.target_list
    else:
        warnings.warn(
            f'Data path differs from that in the stroke_config file; to reduce the amount of time spent loading '
            f'files, modify stroke_config.py: data_path = {path}')
        testing_dir = os.path.join(path, stroke_config.testing['dir_name'])
        bids_loader_test = BIDSLoader(root_dir=testing_dir,
                                      data_entities=[{'subject': '',
                                                      'session': '',
                                                      'desc': 'T1FinalResampledNorm'}],
                                      target_entities=[{'label': 'L',
                                                        'desc': 'T1lesion',
                                                        'suffix': 'mask'}],
                                      data_derivatives_names=['ATLAS'],
                                      target_derivatives_names=['ATLAS'],
                                      label_names=['not lesion', 'lesion'],
                                      batch_size=stroke_config.testing['batch_size'])

    if(stroke_config.is_quick_test):
        return bids_loader_test.data_list[:stroke_config.num_subjects_quick_test], \
               bids_loader_test.target_list[:stroke_config.num_subjects_quick_test]
    else:
        return bids_loader_test.data_list, bids_loader_test.target_list
