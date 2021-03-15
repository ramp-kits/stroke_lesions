import numpy as np
from sklearn.base import BaseEstimator
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Conv3D
from keras import Model
from keras.optimizers import Adam
from rampwf.workflows.image_classifier import get_nb_minibatches
from sklearn.pipeline import Pipeline
from nilearn.image import load_img
from joblib import Memory

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
        print(img_index)
        if self.y is not None:
            return img, self.y[img_index]
        else:
            return img


class KerasSegmentationClassifier(BaseEstimator):
    def __init__(self, image_size):
        self.batch_size = 6
        self.xdim, self.ydim, self.zdim = image_size
        self.model = self.model_simple()

    def _build_train_generator(self, img_loader, indices, batch_size=1,
                               shuffle=False):
        indices = indices.copy()

        nb = len(indices)
        X = np.zeros((batch_size, self.xdim, self.ydim, self.zdim, 1))
        Y = np.zeros((batch_size, self.xdim, self.ydim, self.zdim, 1))

        while True:
            if shuffle:
                np.random.shuffle(indices)
            for start in range(0, nb, batch_size):
                stop = min(start + batch_size, nb)
                # load the next minibatch in memory.
                # The size of the minibatch is (stop - start),
                # which is `batch_size` for the all except the last
                # minibatch, which can either be `batch_size` if
                # `nb` is a multiple of `batch_size`, or `nb % batch_size`.
                bs = stop - start
                for i, img_index in enumerate(indices[start:stop]):
                    x, y = img_loader.load(img_index)
                    X[i] = x[:, :, :, np.newaxis]
                    Y[i] = y[:, :, :, np.newaxis]
                yield X[:bs], Y[:bs]

    def _build_test_generator(self, img_loader, batch_size=1):
        X = np.zeros((batch_size, self.xdim, self.ydim, self.zdim, 1))
        nb = img_loader.n_paths

        for start in range(0, nb, batch_size):
            stop = min(start + batch_size, nb)
            bs = stop - start
            for i, img_index in enumerate(range(start, stop)):
                x = img_loader.load(img_index)
                X[i] = x[:, :, :, np.newaxis]
            yield X[:bs]

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

        gen_train = self._build_train_generator(
            img_loader,
            indices=ind_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        gen_valid = self._build_train_generator(
            img_loader,
            indices=ind_valid,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.model.fit(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_train, self.batch_size),
            epochs=100,
            max_queue_size=1,
            workers=0,
            use_multiprocessing=False,
            validation_data=gen_valid,
            validation_steps=get_nb_minibatches(nb_valid, self.batch_size),
            verbose=1
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
        gen_test = self._build_test_generator(img_loader, self.batch_size)

        y_pred = self.model.predict(
            gen_test,
            batch_size=1
        )
        y_pred = (y_pred > 0.5) * 1
        # remove the last dim
        return y_pred[..., 0]


def get_estimator():
    image_size = (197, 233, 189)
    # initiate a deep learning algorithm
    deep = KerasSegmentationClassifier(image_size)

    pipeline = Pipeline([
        ('classifier', deep)
    ])

    return pipeline
