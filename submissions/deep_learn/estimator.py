import numpy as np
import os
from sklearn.base import BaseEstimator
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling3D, UpSampling3D
import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose  # , Deconvolution3D
from keras_contrib.layers.normalization.instancenormalization import \
    InstanceNormalization
from keras.layers import Activation
from keras.layers import Input, Conv3D
from keras.layers.merge import concatenate
from keras.layers import Concatenate
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from multiprocessing import cpu_count

from sklearn.pipeline import Pipeline
from nilearn.image import load_img
from joblib import Memory

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
K.set_image_data_format('channels_first')

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
        # self.model = self.model_simple()
        self.model = self.unet_model_3d()
        # self.model = self.unet_simple()

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
                        Y[i] = y[:self.xdim,
                                 :self.ydim,
                                 :self.zdim,
                                 np.newaxis]
                    else:
                        go_on = False
                        x = img_loader.load(img_index)
                    X[i] = x[:self.xdim,
                             :self.ydim,
                             :self.zdim,
                             np.newaxis]

                if train:
                    print(f'x shape: {X.shape}, y shape: {Y.shape}')
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

    def unet_simple(self):
        # define a simple model
        inputs = Input((self.xdim, self.ydim, self.zdim, 1))
        x = BatchNormalization()(inputs)
        # downsampling
        down1conv1 = Conv3D(2, (3, 3, 3), activation='relu',
                            padding='same')(x)
        down1conv1 = Conv3D(2, (3, 3, 3), activation='relu',
                            padding='same')(down1conv1)
        down1pool = MaxPooling3D((2, 2, 2))(down1conv1)
        # middle
        mid_conv1 = Conv3D(2, (3, 3, 3), activation='relu',
                           padding='same')(down1pool)
        mid_conv1 = Conv3D(2, (3, 3, 3), activation='relu',
                           padding='same')(mid_conv1)

        # upsampling
        up1deconv = Conv3DTranspose(2, (3, 3, 3), strides=(2, 2, 2),
                                    activation='relu')(mid_conv1)
        up1concat = Concatenate()([up1deconv, down1conv1])
        up1conv1 = Conv3D(2, (3, 3, 3), activation='relu',
                          padding='same')(up1concat)
        up1conv1 = Conv3D(2, (3, 3, 3), activation='relu',
                          padding='same')(up1conv1)
        output = Conv3D(1, (3, 3, 3), activation='softmax',
                        padding='same')(up1conv1)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(lr=self.initial_learning_rate),
                      loss=_dice_coefficient_loss)

        return model

    def model_simple(self):
        # define a simple model
        inputs = Input((self.xdim, self.ydim, self.zdim, 1))
        down1conv1 = Conv3D(32, (6, 6, 6), activation='relu',
                            padding='same')(inputs)
        batch_norm = BatchNormalization()(down1conv1)
        output = Conv3D(1, (3, 3, 3), activation='sigmoid',
                        padding='same')(batch_norm)
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(lr=self.initial_learning_rate),
                      loss=_dice_coefficient_loss)
        print(model.summary())
        return model

    def unet_model_3d(
            self, pool_size=(2, 2, 2), n_labels=1,
            deconvolution=False,
            depth=4, n_base_filters=16,
            batch_normalization=False, activation_name="sigmoid"):
        """
        Builds the 3D UNet Keras model.f
        :param metrics: List metrics to be calculated during model training
            (default is dice coefficient).
        :param n_base_filters: The number of filters that the first layer
            in the convolution network will have. Following layers will contain
            a multiple of this number. Lowering this number will likely reduce
            the amount of memory required to train the model.
        :param depth: indicates the depth of the U-shape for the model.
            The greater the depth, the more max pooling layers will be added
            to the model. Lowering the depth may reduce the amount of memory
            required for training.
        :param input_shape: Shape of the input data
            (n_chanels, x_size, y_size, z_size). The x, y, and z sizes
            must be divisible by the pool size to the power of the depth
            of the UNet, that is pool_size^depth.
        :param pool_size: Pool size for the max pooling operations.
        :param n_labels: Number of binary labels that the model is learning.
        :param deconvolution: If set to True, will use transpose
            convolution(deconvolution) instead of up-sampling. This increases
            the amount memory required during training.
        :return: Untrained 3D UNet Model
        """
        input_shape = (1, self.xdim, self.ydim, self.zdim)
        inputs = Input(input_shape)
        current_layer = inputs
        levels = list()

        # add levels with max pooling
        for layer_depth in range(depth):
            layer1 = self._create_convolution_block(
                input_layer=current_layer,
                n_filters=n_base_filters*(2**layer_depth),
                batch_normalization=batch_normalization
                )
            layer2 = self._create_convolution_block(
                input_layer=layer1,
                n_filters=n_base_filters*(2**layer_depth)*2,
                batch_normalization=batch_normalization
                )
            if layer_depth < depth - 1:
                current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])
        # add levels with up-convolution or up-sampling
        for layer_depth in range(depth-2, -1, -1):
            up_convolution = self._get_up_convolution(
                pool_size=pool_size,
                deconvolution=deconvolution,
                n_filters=current_layer.get_shape()[1]
                )(current_layer)
            concat = concatenate(
                [up_convolution, levels[layer_depth][1]],
                axis=1
                )
            current_layer = self._create_convolution_block(
                n_filters=levels[layer_depth][1].get_shape()[1],
                input_layer=concat,
                batch_normalization=batch_normalization
                )
            current_layer = self._create_convolution_block(
                n_filters=levels[layer_depth][1].get_shape()[1],
                input_layer=current_layer,
                batch_normalization=batch_normalization
                )

        final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
        act = Activation(activation_name)(final_convolution)
        model = Model(inputs=inputs, outputs=act)

        model.compile(optimizer=Adam(lr=self.initial_learning_rate),
                      loss=_dice_coefficient_loss)
        print(model.summary())
        return model

    def _create_convolution_block(
            self, input_layer, n_filters, batch_normalization=False,
            kernel=(3, 3, 3), activation=None,
            padding='same', strides=(1, 1, 1),
            instance_normalization=False):
        """
        :param strides:
        :param input_layer:
        :param n_filters:
        :param batch_normalization:
        :param kernel:
        :param activation: Keras activation layer to use. (default is 'relu')
        :param padding:
        :return:
        """
        layer = Conv3D(
            n_filters, kernel, padding=padding, strides=strides
            )(input_layer)
        if batch_normalization:
            layer = BatchNormalization(axis=1)(layer)
        elif instance_normalization:
            layer = InstanceNormalization(axis=1)(layer)
        if activation is None:
            return Activation('relu')(layer)
        else:
            return activation()(layer)

    def _compute_level_output_shape(
            self, n_filters, depth, pool_size, image_shape):
        """
        Each level has a particular output shape based on the number of filters
        used in that level and the depth or number
        of max pooling operations that have been done on the data at that
        point.
        :param image_shape: shape of the 3d image.
        :param pool_size: the pool_size parameter used in the max pooling
            operation.
        :param n_filters: Number of filters used by the last node in a given
        level.
        :param depth: The number of levels down in the U-shaped model a given
            node is.
        :return: 5D vector of the shape of the output node
        """
        output_image_shape = np.asarray(
            np.divide(image_shape,
                      np.power(pool_size, depth)
                      ), dtype=np.int32).tolist()
        return tuple([None, n_filters] + output_image_shape)

    def _get_up_convolution(self, n_filters, pool_size, kernel_size=(2, 2, 2),
                            strides=(2, 2, 2), deconvolution=False):
        #if deconvolution:
        #    kernel_size = strides = pool_size
        #    return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
        #                           strides=strides)
        #else:
        return UpSampling3D(size=pool_size)

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
    # image_size = (197, 233, 189)
    image_size = (192, 224, 176)
    epochs = 150
    batch_size = 1
    initial_learning_rate = 0.01
    learning_rate_drop = 0.5
    learning_rate_patience = 5
    early_stopping_patience = 10
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
