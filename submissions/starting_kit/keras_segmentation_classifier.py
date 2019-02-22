import numpy as np
import scipy.ndimage as nd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from keras.layers import Input, MaxPooling3D, UpSampling3D, Conv3D, Reshape, Conv3DTranspose


class KerasSegmentationClassifier(BaseEstimator):
    def __init__(self):
        #self.clf = RandomForestClassifier(n_estimators=3, max_depth=10)
        self.clf = DummyClassifier(strategy="constant", constant=0)
        #self.shift = 2 # how many neighbours are taken for calculating features       
        self.batch_size = 1
    
    def _build_train_generator(self, img_loader, indices, batch_size,
                               shuffle=False):
        indices = indices.copy()
        print('indices: ',(indices))
        nb = len(indices)
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
                Y[:] = 0
                for i, img_index in enumerate(indices[start:stop]):
                    x, y = img_loader.load(img_index)
                    x = self._transform(x)
                    X[i] = x
                    Y[i, y] = 1
                    yield X[:bs], Y[:bs]
        '''
        def get_unet(inputs, n_classes):

            x = BatchNormalization()(inputs)
            
            conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
            conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
            conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

            conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
            conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

            up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
            conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
            conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

            up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
            conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
            conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

            up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
            conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
            conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

            up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
            conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
            conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

            conv10 = Conv2D(n_classes, (1, 1), activation='linear')(conv9)
       
    
    return conv10
    '''   

    def fit(self, img_loader):
        # takes imaage ravels it and returns 1s where there are maxs
        #return self
        nb = len(img_loader)
        nb_train = int(nb * 0.9)
        valid = nb - nb_train

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
        print('fitting')

        '''
        gen_valid = self._build_train_generator(
            img_loader,
            indices=ind_valid,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.model.fit_generator(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_train, self.batch_size),
            epochs=1,
            max_queue_size=16,
            workers=1,
            use_multiprocessing=True,
            validation_data=gen_valid,
            validation_steps=get_nb_minibatches(nb_valid, self.batch_size),
            verbose=1
        )
        '''





    def predict(self, X):
        # X_features = self._get_features_scipy(X)
        #X = X.ravel()[:, np.newaxis]
        
        #return self.clf.predict(X)
        # takes image ravels it and returns 1s where there are maxs
        #X = X.ravel() #[:, np.newaxis]        
        thres = np.max(X)
        X[X < thres] = 0
        X[X == thres] = 1
        return X.astype(np.uint8)


