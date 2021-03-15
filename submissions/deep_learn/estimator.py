import numpy as np
import os
from sklearn.base import BaseEstimator
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Conv3D
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from multiprocessing import cpu_count
import tensorflow as tf

from sklearn.pipeline import Pipeline
from nilearn.image import load_img
from joblib import Memory

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

gpus = tf.config.experimental.list_physical_devices('GPU')
# avoid allocating the full GPU memory
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

mem = Memory('.')


@mem.cache
def load_img_data(fname):
    return load_img(fname).get_fdata()


def _dice_coefficient_loss(y_true, y_pred):
    return -_dice_coefficient(y_true, y_pred)


def _dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return ((2. * intersection + smooth) / (K.sum(y_true_f) +
            K.sum(y_pred_f) + smooth))


# Using the generator pattern (an iterable)
class ImageLoader():

    def __init__(self, X_paths, y=None):
        self.X_paths = X_paths
        self.n_paths = len(X_paths)
        self.y = y

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def load(self, img_index):
        img = load_img_data(self.X_paths[img_index])
        if self.y is not None:
            return img, self.y[img_index]
        else:
            return img


class KerasSegmentationClassifier(BaseEstimator):
    def __init__(self, image_size, epochs=100, initial_learning_rate=0.01,
                 learning_rate_patience=10, early_stopping_patience=50,
                 learning_rate_drop=0.5, batch_size=2, workers=10):
        """
        image_size: tuple with three elements (x, y, z)
            which are the dimensions of the images
        epochs: int,
            cutoff the training after this many epochs
        initial_learning_rate: float,
        learning_rate_patience: float,
            learning rate will be reduced after this many epochs if
            the validation loss is not improving
        early_stopping_patience: float,
            training will be stopped after this many epochs without
            the validation loss improving
        learning_rate_drop: float,
            factor by which the learning rate will be reduced
        batch_size: int
        """
        self.batch_size = batch_size
        self.xdim, self.ydim, self.zdim = image_size
        self.model = self.model_simple()
        self.epochs = epochs
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_drop = learning_rate_drop
        self.learning_rate_patience = learning_rate_patience
        self.early_stopping_patience = early_stopping_patience
        if workers == -1:
            self.workers = cpu_count()
        else:
            self.workers = workers
        print(f'workers: {self.workers}')

    def _build_generator(self, img_loader, indices=None,
                         train=True, shuffle=False):
        """
        set train to False if you use it for test
        """
        if indices is not None:
            indices = indices.copy()
            nb = len(indices)
        else:
            nb = img_loader.n_paths
            indices = range(nb)

        X = np.zeros((self.batch_size, self.xdim, self.ydim, self.zdim, 1))
        if train:
            Y = np.zeros((self.batch_size, self.xdim, self.ydim, self.zdim, 1))

        go_on = True
        while go_on:
            if shuffle:
                np.random.shuffle(indices)
            for start in range(0, nb, self.batch_size):
                stop = min(start + self.batch_size, nb)
                # load the next minibatch in memory.
                # The size of the minibatch is (stop - start),
                # which is `batch_size` for the all except the last
                # minibatch, which can either be `batch_size` if
                # `nb` is a multiple of `batch_size`, or `nb % batch_size`.
                bs = stop - start
                assert bs <= self.batch_size
                for i, img_index in enumerate(indices[start:stop]):
                    if train:
                        x, y = img_loader.load(img_index)
                        Y[i] = y[:, :, :, np.newaxis]
                    else:
                        x = img_loader.load(img_index)
                        X[i] = x[:, :, :, np.newaxis]
                        go_on = False

                if train:
                    yield X[:bs], Y[:bs]
                else:
                    yield X[:bs]

    def _get_nb_minibatches(self, nb_samples, batch_size):
        """Compute the number of minibatches for keras.

        See [https://keras.io/models/sequential]
        """
        return (nb_samples // batch_size) +\
            (1 if (nb_samples % batch_size) > 0 else 0)

    def _get_callbacks(self, verbosity=1):
        """
        get callbacks for fit
        """
        callbacks = list()
        callbacks.append(
            ReduceLROnPlateau(
                factor=self.learning_rate_drop,
                patience=self.learning_rate_patience,
                verbose=verbosity
                )
            )
        if self.early_stopping_patience:
            callbacks.append(
                EarlyStopping(
                    verbose=verbosity,
                    patience=self.early_stopping_patience
                    )
                )
        return callbacks

    def fit(self, X, y):

        img_loader = ImageLoader(X, y)
        np.random.seed(42)
        nb = len(X)
        nb_train = int(nb * 0.9)
        nb_valid = nb - nb_train

        indices = np.arange(nb)
        np.random.shuffle(indices)

        ind_train = indices[0: nb_train]
        ind_valid = indices[nb_train:]

        gen_train = self._build_generator(
            img_loader,
            indices=ind_train,
            shuffle=True
        )
        gen_valid = self._build_generator(
            img_loader,
            indices=ind_valid,
            shuffle=True
        )
        if self.workers > 1:
            use_multiprocessing = False
        else:
            use_multiprocessing = False
        self.model.fit(
            gen_train,
            steps_per_epoch=self._get_nb_minibatches(
                nb_train, self.batch_size
                ),
            epochs=self.epochs,
            max_queue_size=1,
            use_multiprocessing=use_multiprocessing,
            validation_data=gen_valid,
            validation_steps=self._get_nb_minibatches(
                nb_valid, self.batch_size
                ),
            verbose=1,
            workers=self.workers,
            callbacks=self._get_callbacks()
        )

    def model_simple(self):
        # define a simple model
        inputs = Input((self.xdim, self.ydim, self.zdim, 1))
        down1conv1 = Conv3D(32, (6, 6, 6), activation='relu',
                            padding='same')(inputs)
        batch_norm = BatchNormalization()(down1conv1)
        output = Conv3D(1, (3, 3, 3), activation='sigmoid',
                        padding='same')(batch_norm)
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(lr=0.1),  # 'rmsprop',
                      loss=_dice_coefficient_loss,  # 'mean_squared_error',
                      metrics=[_dice_coefficient])
        print(model.summary())
        return model

    def predict(self, X):
        img_loader = ImageLoader(X)
        gen_test = self._build_generator(img_loader, train=False)

        y_pred = self.model.predict(
            gen_test,
            batch_size=1
        )
        # threshold the data on 0.5; return only 1s and 0s in y_pred
        y_pred = (y_pred > 0.5) * 1
        # remove the last dim
        return y_pred[..., 0]


def get_estimator():
    image_size = (197, 233, 189)
    epochs = 100
    batch_size = 2
    initial_learning_rate = 0.01
    learning_rate_drop = 0.5
    learning_rate_patience = 10
    early_stopping_patience = 50
    workers = 1  # -1 if you want to use all available CPUs

    # initiate a deep learning algorithm
    deep = KerasSegmentationClassifier(
        image_size, epochs=epochs, batch_size=batch_size,
        initial_learning_rate=initial_learning_rate,
        learning_rate_drop=learning_rate_drop,
        learning_rate_patience=learning_rate_patience,
        early_stopping_patience=early_stopping_patience,
        workers=workers
        )

    pipeline = Pipeline([
        ('classifier', deep)
    ])

    return pipeline
