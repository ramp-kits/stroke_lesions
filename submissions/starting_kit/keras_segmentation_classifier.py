import numpy as np
import scipy.ndimage as nd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
	
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
#from keras.layers import Input, MaxPooling3D, UpSampling3D, Conv3D, Reshape, Conv3DTranspose
import os
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose
from keras.layers import Concatenate
from keras import Model
from rampwf.workflows.image_classifier import get_nb_minibatches

class KerasSegmentationClassifier(BaseEstimator):
    def __init__(self):
        self.clf = DummyClassifier(strategy="constant", constant=0)
        self.batch_size = 1
        self.model = self.model_simple()

    def _build_train_generator(self, img_loader, indices, batch_size=1,
                               shuffle=False):
        print('indices: {}'.format(indices))
        indices = indices.copy()
        
        nb = len(indices)
        X = np.zeros((batch_size, 197, 233, 189, 1))
        Y = np.zeros((batch_size, 197, 233, 189, 1))
        #X = np.zeros((batch_size, 32, 32, 3))
        #Y = np.zeros((batch_size, 403))
        while True:
            if shuffle:
                np.random.shuffle(indices)
            # exchange for unet Keras
            for start in range(0, nb, batch_size):
                stop = min(start + batch_size, nb)
                # load the next minibatch in memory.
                # The size of the minibatch is (stop - start),
                # which is `batch_size` for the all except the last
                # minibatch, which can either be `batch_size` if
                # `nb` is a multiple of `batch_size`, or `nb % batch_size`.
                bs = stop - start
                #Y[:] = 0
                for i, img_index in enumerate(indices[start:stop]):
                    x, y = img_loader.load(img_index)
                    #x = self._transform(x)
                    X[i] = x[:,:,:,np.newaxis]
                    Y[i] = y[:,:,:,np.newaxis]
                    yield X[:bs], Y[:bs]

    def _build_test_generator(self, img_loader, batch_size=1):
        nb = len(img_loader)

        X = np.zeros((batch_size, 197, 233, 189, 1))
        Y = np.zeros((batch_size, 197, 233, 189, 1))

        while True:
            for start in range(0, nb, batch_size):
                stop = min(start + batch_size, nb)
                # load the next minibatch in memory.
                # The size of the minibatch is (stop - start),
                # which is `batch_size` for the all except the last
                # minibatch, which can either be `batch_size` if
                # `nb` is a multiple of `batch_size`, or `nb % batch_size`.
                bs = stop - start
                for i, img_index in enumerate(range(start, stop)):
                    x = img_loader.load(img_index)
                    X[i] = x[:,:,:,np.newaxis]
                yield X[:bs]


    def fit(self, img_loader):
        # takes imaage ravels it and returns 1s where there are maxs
        #return self
        np.random.seed(24)
        nb = len(img_loader)
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
        '''
        # make sure that the memory error does not come up in generators
        for idx, i in enumerate(gen_train):
            print('train')
            iter(gen_train)
            print('valid')
            iter(gen_valid)
            print(idx)

        '''
        self.model.fit_generator(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_train, self.batch_size),
            epochs=1,
            max_queue_size=3,
            workers=1,
            use_multiprocessing=True,
            validation_data=gen_valid,
            validation_steps=get_nb_minibatches(nb_valid, self.batch_size),
            verbose=1
        )
        
        

    def model_simple(self):
            
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

        '''
        train_suffix='_LesionSmooth_*.nii.gz'
        train_id = get_train_data(path='.')
        brain_image = _read_brain_image('.', train_id[1])
        mask = _read_stroke_segmentation('.', train_id[1]) 

        model.fit(brain_image[None, ..., None], mask[None, ..., None].astype(bool))
        #model.fit_on_batch
        '''
        return model

    def predict(self, img_loader):
        # X_features = self._get_features_scipy(X)
        #X = X.ravel()[:, np.newaxis]
        
        #return self.clf.predict(X)
        # takes image ravels it and returns 1s where there are maxs
        #X = X.ravel() #[:, np.newaxis]        
        thres = np.max(X)
        X[X < thres] = 0
        X[X == thres] = 1
        return X.astype(np.uint8)

        nb_test = len(img_loader)
        gen_test = self._build_test_generator(img_loader, self.batch_size)
        return self.model.predict(
            gen_test,
            batch_size=1
            #steps=get_nb_minibatches(nb_test, self.batch_size),
            #max_queue_size=16,
            #workers=1,
            #use_multiprocessing=True,
            #verbose=0
        )
        # predict(x, batch_size=None, verbose=0, steps=None, callbacks=None)



