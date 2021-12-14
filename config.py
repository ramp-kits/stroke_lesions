from bids_loader import BIDSLoader
from os.path import join
import os
from bids.exceptions import BIDSValidationError

estimator_filename = 'estimator.py'
data_path = 'data/'
# Used for checking that the data was downloaded correctly:
encrypted_hash = '417c77c9d6dbd2fbdaecc70c33516299f579ad79fca8cebbff344255a53dbf67'

training = {'batch_size':               5,
            'dir_name':                join(data_path, 'train'),
            'data_entities':            [{'subject': '',
                                          'session': '',
                                          'suffix': 'T1w'}],
            'target_entities':          [{'label': 'L',
                                          'desc': 'T1lesion',
                                          'suffix': 'mask'}],
            'data_derivatives_names':   ['ATLAS'],
            'target_derivatives_names': ['ATLAS'],
            'label_names':              ['not lesion', 'lesion']}

cross_validation = {'n_splits': 5,
                    'train_size': 0.6,
                    'random_state': 9001}

testing = {'dir_name':                 join(data_path, 'test'),
           'batch_size':                training['batch_size'],
           'test_dir_name':             'test',
           'data_entities':             [{'subject': '',
                                          'session': '',
                                          'suffix': 'T1w'}],
           'target_entities':           [{'label': 'L',
                                          'desc': 'T1lesion',
                                          'suffix': 'mask'}],
           'data_derivatives_names':    ['ATLAS'],
           'target_derivatives_names':  ['ATLAS'],
           'label_names':               ['not lesion', 'lesion']}


try:
    if(not os.path.exists(training['dir_name'])):
        train_path = join(os.path.dirname(__file__), data_path, 'train')
        if(os.path.exists(train_path)):
            print(f'Changing training data path to {train_path}')
            training['dir_name'] = train_path

    bids_loader_train = BIDSLoader(root_dir=training['dir_name'],
                                   data_entities=training['data_entities'],
                                   target_entities=training['target_entities'],
                                   data_derivatives_names=training['data_derivatives_names'],
                                   target_derivatives_names=training['target_derivatives_names'],
                                   label_names=training['label_names'],
                                   batch_size=training['batch_size'])
except BIDSValidationError:
    print('Warning: BIDS default path is not valid. Consider modifying config.py to match your data structure.')
    # default training path not valid; ignore
    pass

try:
    if (not os.path.exists(testing['dir_name'])):
        test_path = join(os.path.dirname(__file__), data_path, 'test')
        if (os.path.exists(test_path)):
            print(f'Changing training data path to {test_path}')
            testing['dir_name'] = test_path
    bids_loader_test = BIDSLoader(root_dir=testing['dir_name'],
                                  data_entities=testing['data_entities'],
                                  target_entities=testing['target_entities'],
                                  data_derivatives_names=testing['data_derivatives_names'],
                                  target_derivatives_names=testing['target_derivatives_names'],
                                  label_names=testing['label_names'],
                                  batch_size=testing['batch_size'])
except BIDSValidationError:
    print('Warning: BIDS default path is not valid. Consider modifying config.py to match your data structure.')
    # default testing path not valid; ignore
    pass