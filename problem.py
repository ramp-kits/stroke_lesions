import bids
bids.config.set_option('extension_initial_dot', True)  # bids warning suppression

import rampwf as wf
import torch as T
import sklearn
import prediction
from bids_workflow import BIDSWorkflow
from rampwf.prediction_types.base import BasePrediction

from scoring import DiceCoeff
import os
import numpy as np

import config
from bids_loader import BIDSLoader
import warnings

problem_title = "ATLAS Stroke Lesion Segmentation"

workflow = BIDSWorkflow()  # Define workflow; this determines how data is trained + tested.
Predictions = prediction.BIDSPrediction  # Class containing data + targets
score_types = [DiceCoeff()]  # Scores to evaluate; object is instantiated because RAMP expects some fields to be defined


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
    strat = sklearn.model_selection.ShuffleSplit(n_splits=config.cross_validation['n_splits'],
                                                 train_size=config.cross_validation['train_size'],
                                                 random_state=config.cross_validation['random_state'])
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
    # Otherwise; continue, but warn user and give instructions on how to optimize settings.
    if(path == '.' or path == './'):
        path = 'data'
    if(os.path.abspath(path) == os.path.abspath(config.data_path)):
        return config.bids_loader_train.data_list, config.bids_loader_train.target_list
    else:
        warnings.warn(f'Data path differs from that in the config file; to reduce the amount of time spent loading '
                      f'files, modify config.py: data_path = {path}')
        training_dir = os.path.join(path, config.training['dir_name'])
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
                                       batch_size=config.training['batch_size'])
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
    # Otherwise; continue, but warn user and give instructions on how to optimize settings.
    if (path == '.' or path == './'):
        path = 'data'
    if (os.path.abspath(path) == os.path.abspath(config.data_path)):
        return config.bids_loader_test.data_list, config.bids_loader_test.target_list
    else:
        warnings.warn(f'Data path differs from that in the config file; to reduce the amount of time spent loading '
                      f'files, modify config.py: data_path = {path}')
        testing_dir = os.path.join(path, config.testing['dir_name'])
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
                                       batch_size=config.testing['batch_size'])

    return bids_loader_test.data_list, bids_loader_test.target_list
