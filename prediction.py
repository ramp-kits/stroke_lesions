from bids_loader import BIDSLoader
import numpy as np
from rampwf.prediction_types.base import BasePrediction

import bids
bids.config.set_option(
    'extension_initial_dot',
    True)  # bids warning suppression


class BIDSPrediction(BasePrediction):
    def __init__(self,
                 label_names: list = None,
                 y_pred: list = None,
                 y_true: list = None,
                 fold_is: list = None,
                 n_samples: int = None):
        '''
        RAMP prediction class for BIDS dataset
        Parameters
        ----------
        label_names : list [str]
            List of names for the target labels.
        y_pred : list
        '''

        self.label_names = label_names
        if(y_pred is not None):
            if(fold_is is not None):
                y_pred = [y_pred[i] for i in fold_is]
            self.y_pred = y_pred
        else:
            self.y_pred = []
        if(y_true is not None):
            if(fold_is is not None):
                y_true = [y_true[i] for i in fold_is]
            self.y_true = np.array([BIDSLoader.load_image_tuple(y)
                                   for y in y_true], dtype=bool)
        else:
            self.y_true = []

        if(y_pred is None and y_true is None):
            if(n_samples is None):
                raise ValueError(
                    'Either y_pred, y_true, or n_samples must be defined')
            else:
                self.y_pred = [np.nan for _ in range(n_samples)]

        return

    def __str__(self):
        return f'y_pred: {len(self.y_pred)}\n y_true: {len(self.y_true)}'

    def set_valid_in_train(self,
                           predictions: list,
                           test_is: list):
        '''
        Sets self.y_pred to predictions; the position in self.y_pred is determined by the element in test_is in the
        same position as the prediction (i.e., self.y_pred[test_idx] = predictions.

        Parameters
        ----------
        predictions : list
            List of values for prediction. For BIDSPrediction, this is a tuple containing the estimator and the file
            for which to make the prediction.
        test_is : list
            List of indices to determine which position in self.y_pred to place the predictions.

        Returns
        -------
        None
        '''
        while(np.max(test_is) >= len(self.y_pred)):
            self.y_pred.append(np.nan)
        for i, pred in zip(test_is, predictions.y_pred):
            self.y_pred[i] = pred
        return

    @property
    def valid_indexes(self):
        '''
        Returns a list of boolean with True for non-NaN elements of self.y_pred, and False for NaN elements.
        Returns
        -------
        list [bool]
            List of booleans indicating whether the element
        '''
        is_nan = np.zeros((len(self.y_pred)), dtype=bool)
        for idx, pred in enumerate(self.y_pred):
            if(isinstance(pred, float) and np.isnan(pred)):
                is_nan[idx] = 1
        return ~is_nan

    def set_slice(self, valid_indexes):
        # This immediately returns to prevent RAMP from doing bagging; the current structure doesn't work with RAMP's
        # bagging.
        return

    @classmethod
    def combine(cls,
                predictions_list: list):
        '''
        Combines the y_pred and y_true of the Predictions in preditions_list into a single Prediction. Label names
        are taken from the first element.
        Parameters
        ----------
        predictions_list : list [Prediction]
            List of BIDSPrediction to combine.

        Returns
        -------
        BIDSPrediction
            BIDSPrediction with y_pred and y_true merged from the input.
        '''
        label_names = predictions_list[0].label_names
        pred_list = []
        true_list = []
        for p in predictions_list:
            pred_list += p.y_pred
            true_list += p.y_true
        second_pred = []
        second_true = []
        # Remove NaN from list
        for p, t in zip(pred_list, true_list):
            if(isinstance(p, float) and np.isnan(p)):
                continue
            second_pred.append(p)
            second_true.append(t)
        new_prediction = cls(label_names=label_names,
                             y_pred=second_pred,
                             y_true=second_true)
        return new_prediction
