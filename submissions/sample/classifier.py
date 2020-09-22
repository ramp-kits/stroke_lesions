
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.multioutput import MultiOutputClassifier
from nilearn.image import load_img
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin


N_JOBS = 1

class ImageLoader(object):
    """
    Load and image and optionally its segmented mask.
    In segmentation_classifier.py, both `fit` and `predict` take as input
    an instance of `ImageLoader`.
    ImageLoader is used in `fit` and `predict` to either load one 3d image
    and its corresponding segmented mask  (at training time), or one 3d image
    (at test time).
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
        name of the .csv file, with Img id, it's corresponding patient Id
        (INDI Subject ID) and where this particular data stored
    n_classes : int
        Total number of classes.
    """

    def __init__(self, base_dir, list_subj_dirs, file_name='T1.nii.gz'):
        """ dir_name: 'train' or 'test' """
        self.base_dir = base_dir
        self.list_subj_dirs = list_subj_dirs

        self.file_name = file_name

    def load(self, path_patient):
        """
        Load one image and its corresponding segmented mask (at training time),
        or one image (at test time).
        Parameters
        ==========
        path : string
            path to the image to load
        Returns
        =======
        either a tuple `(x, y)` or `x`, where:
            - x is a numpy array of shape (height, width, depth),
              and corresponds to the image of the requested `index`.
            - y is a numpy array of the same shape as x, however filled only
                with integers of n_classes
        At training time, `y` is given, and `load` returns
        a tuple (x, y).
        At test time, `y` is `None`, and `load` returns `x`.
        """
        if not os.path.exists(path_patient):
            raise IndexError(f"{path_patient} does not exist")

        path_file = os.path.join(path_patient, self.file_name)
        data = load_img(path_file).get_data()

        return data

    def __iter__(self):
        for subj_dir in self.list_subj_dirs:
            subj_dir_load = os.path.join(self.base_dir, subj_dir)
            # do we also want to be able to load batches of data?
            # from: https://pytorch.org/docs/stable/data.html
            # for indices in batch_sampler:
            #    yield collate_fn([dataset[i] for i in indices])

            yield self.load(subj_dir_load)


class Classifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        # returns y filled with only 1s

        x_path = X[0]
        x_data = load_img(x_path)
        x_shape = x_data.shape
        y = np.ones((len(X), x_shape[0], x_shape[1], x_shape[2]))

        return y


def get_estimator():
    dummy = Classifier()

    return dummy