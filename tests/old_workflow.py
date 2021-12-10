import os

import numpy as np

from ..utils.importing import import_module_from_source


class ImageClassifier(object):
    """
    ImageClassifier workflow.
    This workflow is used to train image classification tasks, typically when
    the dataset cannot be stored in memory.
    Submissions need to contain two files, which by default are named:
    image_preprocessor.py and batch_classifier.py (they can be modified
    by changing `workflow_element_names`).
    image_preprocessor.py needs a `tranform` function, which
    is used for preprocessing the images. It takes an image as input
    and it returns an image as an output. Optionally, image_preprocessor.py
    can also have a function `transform_test`, which is used only to preprocess
    images at test time. Otherwise, if `transform_test` does not exist,
    `transform` is used at train and test time.
    batch_classifier.py needs a `BatchClassifier` class, which implements
    `fit` and `predict_proba`, where `fit` takes as input an instance
    of `BatchGeneratorBuilder`.
    Parameters
    ==========
    test_batch_size : int
        batch size used for testing.
    chunk_size : int
        size of the chunk used to load data from disk into memory.
        (see at the top of the file what a chunk is and its difference
         with the mini-batch size of neural nets).
    n_jobs : int
        the number of jobs used to load images from disk to memory as `chunks`.
    n_classes : int
        Total number of classes.
    """

    def __init__(self, test_batch_size, chunk_size, n_jobs, n_classes,
                 workflow_element_names=[
                     'image_preprocessor', 'batch_classifier']):
        self.element_names = workflow_element_names
        self.test_batch_size = test_batch_size
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
        self.n_classes = n_classes

    def train_submission(self, module_path, folder_X_array, y_array,
                         train_is=None):
        """Train a batch image classifier.
        module_path : str
            module where the submission is. the folder of the module
            have to contain batch_classifier.py and image_preprocessor.py.
        X_array : ArrayContainer vector of int
            vector of image IDs to train on
            (it is named X_array to be coherent with the current API,
             but as said here, it does not represent the data itself,
             only image IDs).
        y_array : vector of int
            vector of image labels corresponding to X_train
        train_is : vector of int
           indices from X_array to train on
        """
        folder, X_array = folder_X_array
        if train_is is None:
            train_is = slice(None, None, None)
        image_preprocessor = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=True
        )
        transform_img = image_preprocessor.transform
        transform_test_img = getattr(image_preprocessor,
                                     'transform_test',
                                     transform_img)
        batch_classifier = import_module_from_source(
            os.path.join(module_path, self.element_names[1] + '.py'),
            self.element_names[1],
            sanitize=True
        )
        clf = batch_classifier.BatchClassifier()

        gen_builder = BatchGeneratorBuilder(
            X_array[train_is], y_array[train_is],
            transform_img, transform_test_img,
            folder=folder,
            chunk_size=self.chunk_size, n_classes=self.n_classes,
            n_jobs=self.n_jobs)
        clf.fit(gen_builder)
        return transform_img, transform_test_img, clf

    def test_submission(self, trained_model, folder_X_array):
        """Train a batch image classifier.
        trained_model : tuple (function, Classifier)
            tuple of a trained model returned by `train_submission`.
        X_array : ArrayContainer of int
            vector of image IDs to test on.
            (it is named X_array to be coherent with the current API,
             but as said here, it does not represent the data itself,
             only image IDs).
        """
        folder, X_array = folder_X_array
        transform_img, transform_test_img, clf = trained_model
        it = _chunk_iterator(
            X_array, folder=folder, chunk_size=self.chunk_size)
        y_proba = []
        for X in it:
            for i in range(0, len(X), self.test_batch_size):
                # 1) Preprocessing
                X_batch = X[i: i + self.test_batch_size]
                # X_batch = Parallel(n_jobs=self.n_jobs, backend='threading')(
                #     delayed(transform_img)(x) for x in X_batch)
                X_batch = [transform_test_img(x) for x in X_batch]
                # X is a list of numpy arrays at this point, convert it to a
                # single numpy array.
                try:
                    X_batch = [x[np.newaxis, :, :, :] for x in X_batch]
                except IndexError:
                    # single channel
                    X_batch = [
                        x[np.newaxis, np.newaxis, :, :] for x in X_batch]
                X_batch = np.concatenate(X_batch, axis=0)

                # 2) Prediction
                y_proba_batch = clf.predict_proba(X_batch)
                y_proba.append(y_proba_batch)
        y_proba = np.concatenate(y_proba, axis=0)
        return y_proba