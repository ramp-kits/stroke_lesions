import glob
import os
import numpy as np
import pandas as pd
import rampwf as rw
import scipy.sparse as sps
from nilearn.image import load_img
from sklearn.model_selection import StratifiedShuffleSplit

from rampwf.score_types import BaseScoreType

DATA_HOME = 'data'
RANDOM_STATE = 42

# Author: Maria Telenczuk <https://github.com/maikia>
# License: BSD 3 clause

class ImageLoader(object):
    """
    Load and image and optionally its segmented mask.
    In segmentation_classifier.py, both `fit` and `predict` take as input
    an instance of `ImageLoader`.
    ImageLoader is used in `fit` and `predict` to either load one 3d image
    and its corresponding segmented mask  (at training time), or one 3d image (at test
    time).
    Images are loaded by using the method `load`.
    Parameters
    ==========
    img_ids : ArrayContainer of int
        vector of image IDs to train on
         (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs) corresponding to images from the given .csv file
    folder : str
        folder where the data_file is
    data_file : str
        name of the .csv file, with Img id, it's corresponding patient Id (INDI Subject ID)
        and where this particular data stored
    n_classes : int
        Total number of classes.
    """

    def __init__(self, patient_ids, folder='data/images/', run_type = 'train',
                data_file='ATLAS_Meta-Data_Release_1.1_standard_mni.csv',
                n_classes=[0,1],
                train_suffix='_t1w_deface_stx.nii.gz', test_suffix='_LesionSmooth_*.nii.gz'):
        self.data_file = data_file
        self.folder = folder
        self.patient_ids = patient_ids
        self.n_classes = n_classes
        self.nb_examples = len(patient_ids)
        self.train_suffix = train_suffix
        self.test_suffix = test_suffix
        self.run_type = run_type

    def load(self, index):
        """
        Load one image and its corresponding segmented mask (at training time),
        or one image (at test time).
        Parameters
        ==========
        index : int
            Index of the image to load. it should be one of the img_ids indices
        Returns
        =======
        either a tuple `(x, y)` or `x`, where:
            - x is a numpy array of shape (height, width, depth),
              and corresponds to the image of the requested `index`.
            - y is a numpy array of the same shape as x, however filled only with 
                integers of n_classes
        At training time, `y` is given, and `load` returns
        a tuple (x, y).
        At test time, `y` is `None`, and `load` returns `x`.
        """
        from nilearn.image import load_img
       

        if index < 0 or index >= self.nb_examples:
            raise IndexError("list index out of range")

        #assert np.isin(index, self.img_ids)
        subject_id, path_patient = self._get_patient_path(patient_id=self.patient_ids[index], 
                                            path=self.folder,
                                            data_file=self.data_file)
        
        print('READ BRAIN IMAGES')
        x = self._read_brain_image(subject_id=subject_id, path_patient=path_patient)


        if self.run_type == 'train':
            y = self._read_stroke_segmentation(subject_id=subject_id, path_patient=path_patient)
            print('READ SEGMENTATION')
            return x, y
        elif self.run_type == 'test':
            return x

    def parallel_load(self, indexes, transforms=None):
        """
        Load and image and optionally its mask.
        Load one image and its corresponding mask (at training time),
        or one image (at test time).
        Parameters
        ==========
        index : int
            Index of the image to load.
            It should be within the self.
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
        from nilearn.image import load_img
        from joblib import delayed, Parallel, cpu_count

        # FIXME: check if indexes within self.img_ids

        n_jobs = cpu_count()
        
        xs = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self.load)(index) for index in indexes)
        return xs

    def __iter__(self):
        for i in range(self.nb_examples):
            yield self.load(i)

    def __len__(self):
        return self.nb_examples
            ######################read single data image#################################
    
    def _get_patient_path(self, patient_id, path, data_file):
        print('working on patient: {}'.format(patient_id))
        path_metadata = os.path.join(path, data_file)
        df = pd.read_csv(path_metadata)
        #site_dir = df[df['Img Id'] == img_id]['INDI Site ID']
        #subject = df[df['Img Id'] == img_id]
        
        subject = df[df['INDI Subject ID'] == patient_id]
        if len(subject)>1:
            # FIXME???: for now only first image is taken from a single patient, even if more
            # is given; make sure that it is a correct approach
            try:
                subject = subject[subject['Img Id']==subject.index[0]]
            except:
                import pdb; pdb.set_trace()
        
        assert len(subject) == 1, 'Patient Id >{}< not found'.format(patient_id)

        site_dir = subject['INDI Site ID'].iloc[0]
        subject_id = subject['INDI Subject ID'].iloc[0]
        subject_id_str = self._get_str_subject_id(subject_id)
        session = subject['Session'].iloc[0]
        session=session.replace(" ", "")
        
        path_patient = os.path.join(
            path, site_dir, subject_id_str, session
        )
        return subject_id, path_patient
        
    def _get_str_subject_id(self, subject_id):
        return '{:06d}'.format(subject_id)
    
    def _read_brain_image(self, subject_id, path_patient):
        # the data will be of dimensions: #h,v,d

        path_brain_image = os.path.join(path_patient,
            self._get_str_subject_id(subject_id) + self.train_suffix
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

    def _read_stroke_segmentation(self, subject_id, path_patient):
        #path_patient = _get_patient_path(path, subject_id)
        str_subject_id = self._get_str_subject_id(subject_id)
        
        path_masks = glob.glob(str(path_patient)+'/'+str_subject_id +self.test_suffix) # get all the lesions
        
        mask = self._combine_masks(path_masks)  
        assert np.all(np.isin(mask,self.n_classes)) # mask must have only values as in the self.n_classes
        # convert 3D numpy array to list of 2D sparse matrices
        # mask = [sps.csr_matrix(mask[idx]) for idx in range(np.size(mask,0))]
        return mask  
    
    #######################################################  
        


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
        score = _dice_coeff(y_true_mask, y_pred_mask)

    def _dice_coeff(self, y_true_mask, y_pred_mask)
        if (np.sum(y_pred)==0) & (np.sum(y_true) == 0):
            return 1
        else:
            dice = ((np.sum((y_pred==1)&(y_true==1))*2.0) /
                    (np.sum(y_pred) + np.sum(y_true))
        return dice

# TODO: other ideas:
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
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.Classifier()


# TODO: dice coefficient
# TODO: scoring on the time of calculations?
score_types = [
    rw.score_types.Accuracy(name='acc'),  # sklearn accuracy_score:
# In multilabel classification, this function computes subset accuracy:
# the set of labels predicted for a sample must exactly match the corresponding
# set of labels in y_true.
    rw.score_types.DiceCoeff(),
]

# cross validation
def get_cv(X, y):
    # TODO: correct
    cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=RANDOM_STATE)
    return cv.split(X, y)

def _read_data(path, typ):
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

def get_train_data(path="."):
    return _read_data(path, 'train')

def get_test_data(path="."):
    return _read_data(path, 'test')





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

    # FIXME: return only few first IDs
    if split == 'train':
        return train_id[:15]
    return test_id[:15]

# FIXME: might be better to get Img id instead of Subject Id
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
    

# FIXME: when added to workflow change those lines:
#from ..utils.importing import import_file
import submissions.starting_kit.keras_segmentation_classifier as classifier 

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
             which are to be used for training. 
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
