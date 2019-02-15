from __future__ import division

import submissions.starting_kit.classifier as classifier 
#from ..utils.importing import import_file

import glob
import os
import numpy as np
import pandas as pd
import rampwf as rw
import scipy.sparse as sps
from nilearn.image import load_img
from sklearn.model_selection import StratifiedShuffleSplit


problem_title = 'Stroke segmentation'
#_target_column_name = 'species'
_prediction_label_names = [0, 1]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.Accuracy(name='acc'),
    #rw.score_types.ClassificationError(name='error'),
    #rw.score_types.NegativeLogLikelihood(name='nll'),
    #rw.score_types.F1Above(name='f1_70', threshold=0.7),
]

def dice_coefficient(y_pred, y_true):
    # Dice’s coefficient (DC), which describes the volume overlap between two segmentations 
    # and is sensitive to the lesion size;
    if (np.sum(y_pred)==0) & (np.sum(y_true) == 0):
        return 1
    else:
        dice = (np.sum((y_pred==1)&(y_true==1))*2.0) / (np.sum(y_pred) + np.sum(y_true))
        #print('Dice coefficient is {}'.format(dice))
        return dice

# cross validation
def get_cv(X, y):
    # used from mars 
    # 3 quadrangles for training have not exactly the same size,
    # but for simplicity just cut in 3
    # for each fold use one quadrangle as test set, the other two as training

    # FIXME: k-fold from scikit

    xx = [(np.array([0, 1]), np.array([2])),
          (np.array([1, 2]), np.array([0]))]
    print(xx)
    return xx


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

    # FIXME: return only 7 first IDs 
    if split == 'train':
        return train_id[:3]
    return test_id[:3]

def _get_patient_path(path, subject_id):
    path_metadata = os.path.join(
        path, 'data', 'images', 'ATLAS_Meta-Data_Release_1.1_standard_mni.csv'
    )
    df = pd.read_csv(path_metadata)
    # FIXME: just to take the first occurrence
    site_dir = df[df['INDI Subject ID'] == subject_id]['INDI Site ID'].iloc[0]
    subject_id = _get_str_subject_id(subject_id)
    path_patient = os.path.join(
        path, 'data', 'images', site_dir, subject_id, 't01'
    )
    return path_patient

def _get_str_subject_id(subject_id):
    return '{:06d}'.format(subject_id)

def _read_brain_image(path, subject_id):
    # the data will be of dimensions: #img,h,v,d
    path_patient = _get_patient_path(path, subject_id)
    path_brain_image = os.path.join(path_patient,
        _get_str_subject_id(subject_id) + '_t1w_deface_stx.nii.gz'
    )
    return load_img(path_brain_image).get_data()

def _combine_masks(path_masks):
    mask = load_img(path_masks[0]).get_data().astype(bool)
    for next_mask_path in path_masks[1:]:
        mask2 = load_img(next_mask_path).get_data().astype(bool)
        mask|=mask2
        #mask = np.add(mask, load_img(next_mask_path).get_data())
    mask = mask.astype(np.uint8)
    # masks are to be only 0s for no lesion or 1s for lesion
    #mask[mask > 1] = 1 
    return mask

def _read_stroke_segmentation(path, subject_id):
    # the data will be of dimensions: #img,h,sparse(v,d)
    path_patient = _get_patient_path(path, subject_id)
    path_masks = glob.glob(str(path_patient)+'/*_LesionSmooth_*.nii.gz')
    
    mask = _combine_masks(path_masks)  

    # convert 3D numpy array to list of 2D sparse matrices
    # mask = [sps.csr_matrix(mask[idx]) for idx in range(np.size(mask,0))]
    return mask  

def _read_data(path, split_ids):
    
    X = np.stack([_read_brain_image(path, subject_id)
                   for subject_id in split_ids])

    Y = np.stack([_read_stroke_segmentation(path, subject_id)
                   for subject_id in split_ids])
    print(Y.shape)
    #return X.ravel(), Y.ravel()
    return X, Y

def get_train_data(path='.'):
    # generate the training IDs
    train_id = _read_ids(path, split='train')
    #return _read_data(path, train_id)
    return train_id
    


def get_test_data(path='.'):
    # generate the testing IDs
    test_id = _read_ids(path, split='test')
    #return _read_data(path, test_id)
    return test_id
    

class SimplifiedSegmentationClassifier(object):
    """
    SimplifiedImageClassifier workflow.
    This workflow is used to train image classification tasks, typically when
    the dataset cannot be stored in memory. It is a simplified version
    of the `ImageClassifier` workflow where there is no batch generator
    and no image preprocessor.
    Submissions need to contain one file, which by default by is named
    image_classifier.py (it can be modified by changing
    `workflow_element_names`).
    image_classifier.py needs an `ImageClassifier` class, which implements
    `fit` and `predict_proba`, where both `fit` and `predict_proba` take
    as input an instance of `ImageLoader`.
    Parameters
    ==========
    n_classes : int
        Total number of classes.
    """

    def __init__(self, n_classes=2, workflow_element_names=['segmentation']):
        self.n_classes = n_classes
        #self.element_names = workflow_element_names
        
    ######################read single data image#################################
    def _read_brain_image(self, path, subject_id):
        # the data will be of dimensions: #img,h,v,d
        path_patient = _get_patient_path(path, subject_id)
        path_brain_image = os.path.join(path_patient,
            _get_str_subject_id(subject_id) + '_t1w_deface_stx.nii.gz'
        )
        return load_img(path_brain_image).get_data()
    
    def _combine_masks(self, path_masks):
        mask = load_img(path_masks[0]).get_data().astype(bool)
        for next_mask_path in path_masks[1:]:
            mask2 = load_img(next_mask_path).get_data().astype(bool)
            mask|=mask2
            #mask = np.add(mask, load_img(next_mask_path).get_data())
        mask = mask.astype(np.uint8)
        # masks are to be only 0s for no lesion or 1s for lesion
        #mask[mask > 1] = 1 
        return mask

    def _read_stroke_segmentation(self, path, subject_id):
        # the data will be of dimensions: #img,h,sparse(v,d)
        path_patient = _get_patient_path(path, subject_id)
        path_masks = glob.glob(str(path_patient)+'/*_LesionSmooth_*.nii.gz')
        
        mask = _combine_masks(path_masks)  

        # convert 3D numpy array to list of 2D sparse matrices
        # mask = [sps.csr_matrix(mask[idx]) for idx in range(np.size(mask,0))]
        return mask  
    
    #######################################################  

    def train_submission(self, module_path, patient_idxs): #, y_array):
                         #train_is=None):
        """Train an image classifier.
        module_path : str
            module where the submission is. the folder of the module
            have to contain image_classifier.py.
        X_array : ArrayContainer vector of int
            vector of image IDs to train on
            (it is named X_array to be coherent with the current API,
             but as said here, it does not represent the data itself,
             only image IDs).
        y_array : vector of int
            vector of image labels corresponding to X_train
        #train_is : vector of int
        #   indices from X_array to train on
        """
        #folder, X_array = folder_X_array
        #if train_is is None:
        #    train_is = slice(None, None, None)
        
        
        #image_classifier = import_file(module_path, self.element_names[0])
        
        clf = classifier.Classifier()
        
        # load image one by one
        for patient_id in patient_idxs:
            X = self._read_brain_image(module_path, patient_id)
            y_true = self._read_stroke_segmentation(module_path, patient_id)
            #img_loader = #ImageLoader(
                #idx, #y_array[train_is],
                #folder=folder,
                #n_classes=self.n_classes)
            fitit = clf.fit(X, y_true)
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
        #folder, X_array = folder_X_array
        clf = trained_model
        
        dices = np.zeros(len(patient_idxs))
        # load image one by one
        for idx, patient_id in enumerate(patient_idxs):
            X = self._read_brain_image(module_path, patient_id)
            y_true = self._read_stroke_segmentation(module_path, patient_id)
            #test_img_loader = ImageLoader(
            #    X_array, None,
            #    folder=folder,
            #    n_classes=self.n_classes
            #)
            
            y_pred = clf.predict(X)
            dices[idx] = dice_coefficient(y_pred, y_true)
            #y_proba = clf.predict_proba(test_img_loader)
        return dices


def _image_transform(x, transforms):
    from skimage.transform import rotate
    for t in transforms:
        if t['name'] == 'rotate':
            angle = np.random.random() * (
                t['u_angle'] - t['l_angle']) + t['l_angle']
            rotate(x, angle, preserve_range=True)
    return x


class ImageLoader(object):
    """
    Load and image and optionally its label.
    In image_classifier.py, both `fit` and `predict_proba` take as input
    an instance of `ImageLoader`.
    ImageLoader is used in `fit` and `predict_proba` to either load one image
    and its corresponding label  (at training time), or one image (at test
    time).
    Images are loaded by using the method `load`.
    Parameters
    ==========
    X_array : ArrayContainer of int
        vector of image IDs to train on
         (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs).
    y_array : vector of int or None
        vector of image labels corresponding to `X_array`.
        At test time, it is `None`.
    folder : str
        folder where the images are
    n_classes : int
        Total number of classes.
    """

    def __init__(self, X_array, y_array, folder, n_classes):
        self.X_array = X_array
        self.y_array = y_array
        self.folder = folder
        self.n_classes = n_classes
        self.nb_examples = len(X_array)

    def load(self, index):
        """
        Load and image and optionally its label.
        Load one image and its corresponding label (at training time),
        or one image (at test time).
        Parameters
        ==========
        index : int
            Index of the image to load.
            It should in between 0 and self.nb_examples - 1
        Returns
        =======
        either a tuple `(x, y)` or `x`, where:
            - x is a numpy array of shape (height, width, nb_color_channels),
              and corresponds to the image of the requested `index`.
            - y is an integer, corresponding to the class of `x`.
        At training time, `y_array` is given, and `load` returns
        a tuple (x, y).
        At test time, `y_array` is `None`, and `load` returns `x`.
        """
        from skimage.io import imread

        if index < 0 or index >= self.nb_examples:
            raise IndexError("list index out of range")

        x = self.X_array[index]
        filename = os.path.join(self.folder, '{}'.format(x))
        x = imread(filename)
        if self.y_array is not None:
            y = self.y_array[index]
            return x, y
        else:
            return x

    def parallel_load(self, indexes, transforms=None):
        """
        Load and image and optionally its label.
        Load one image and its corresponding label (at training time),
        or one image (at test time).
        Parameters
        ==========
        index : int
            Index of the image to load.
            It should in between 0 and self.nb_examples - 1
        Returns
        =======
        either a tuple `(x, y)` or `x`, where:
            - x is a numpy array of shape (height, width, nb_color_channels),
              and corresponds to the image of the requested `index`.
            - y is an integer, corresponding to the class of `x`.
        At training time, `y_array` is given, and `load` returns
        a tuple (x, y).
        At test time, `y_array` is `None`, and `load` returns `x`.
        """
        from skimage.io import imread
        from joblib import delayed, Parallel, cpu_count

        for index in indexes:
            assert 0 <= index < self.nb_examples

        n_jobs = cpu_count()
        filenames = [
            os.path.join(self.folder, '{}'.format(self.X_array[index]))
            for index in indexes]
        xs = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(imread)(filename) for filename in filenames)

        if transforms is not None:
            from functools import partial
            transform = partial(_image_transform, transforms=transforms)
            xs = Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(transform)(x) for x in xs)

        if self.y_array is not None:
            ys = [self.y_array[index] for index in indexes]
            return xs, ys
        else:
            return xs

    def __iter__(self):
        for i in range(self.nb_examples):
            yield self.load(i)

    def __len__(self):
        return self.nb_examples
