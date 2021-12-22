from rampwf.utils.importing import import_module_from_source
import os
from stroke import stroke_config
from stroke.bids_loader import BIDSLoader


class BIDSWorkflow():
    def __init__(self,
                 workflow_element_names: list = None,
                 *args, **kwargs):
        '''
        RAMP workflow for training a classifier on BIDS datasets. Its intended use is for estimators applied to
        neuroimaging data that can't typically be stored in memory at once.
        Submissions need to contain one file: estimator.py, with the following requirements:
        - estimator.py  - submitted function
            - class BIDSEstimator  - estimator to train
                - def fit_partial  - defined method for training on portions of data

        Parameters
        ----------
        workflow_element_names : list [str]
            List of the names for the elements of the workflow. Included to be consistent with RAMP API.

        '''
        if(workflow_element_names is None):
            workflow_element_names = ['estimator.py']
        self.element_names = workflow_element_names
        self.estimator = None
        return

    def train_submission(self,
                         module_path: str,
                         X_array: list,
                         y_array: list,
                         train_is: list = None):
        '''
        Trains the submitted estimator.
        Parameters
        ----------
        module_path : str
            Leading path to the user's custom modules (typically submissions/X)
        X_array : list
            List of BIDSImage tuples to load. These correspond to the data. Named X_array to conform to RAMP API.
        y_array : list
            List of BIDSImage tuples to load. These correspond to the labels. Named y_array to conform to RAMP API.
        train_is : list
            List of indices indicating the entries in X_array to use for training.

        Returns
        -------
        estimator
            Trained estimator.
        '''

        if(train_is is None):
            train_is = slice(None, None, None)

        batch_size = stroke_config.training['batch_size']
        estimator_module = import_module_from_source(
            os.path.join(
                module_path,
                self.element_names[0]),
            self.element_names[0],
            sanitize=True)
        self.estimator = estimator_module.BIDSEstimator()

        for idx in range(0, len(train_is), batch_size):
            # Get tuples to load
            data_to_load = [X_array[i] for i in train_is[idx:idx + batch_size]]
            target_to_load = [y_array[i]
                              for i in train_is[idx:idx + batch_size]]
            # Load data
            data = BIDSLoader.load_image_tuple_list(data_to_load)
            target = BIDSLoader.load_image_tuple_list(target_to_load, dtype=stroke_config.data_types['target'])

            # Fit
            self.estimator.fit_partial(data, target)
        return self.estimator

    def test_submission(self,
                        trained_estimator,
                        X_array: list):
        '''
        Returns a list of EstimatorDataPair, which is an object containing .estimator and .pred. Due to the size of each
        prediction (equivalent to the size of the input), we can't return our predictions on the entire set. Instead,
        we're attaching the estimator and the prediction together and returning that. This method's output is sent
        to the scoring function, which expects this structure.
        Parameters
        ----------
        trained_estimator
            Estimator previously trained by train_submission
        X_array : list
            List of BIDSImage tuples on which to make the predictions.

        Returns
        -------
        list [EstimatorDataPair]
            List of EstimatorDataPair containing the trained estimator and the sample on which to make the prediction.
        '''

        preds = []
        for image_tuple in X_array:
            preds.append(EstimatorDataPair(trained_estimator, image_tuple))
        return preds


class EstimatorDataPair():
    def __init__(self, estimator, pred):
        '''
        Holds and estimator and a BIDSImage tuple.
        Parameters
        ----------
        estimator
            Estimator to later use for prediction.
        pred : tuple
            Tuple holding the BIDSImage files to later load and predict on.
        '''
        self.estimator = estimator
        self.pred = pred
        return
