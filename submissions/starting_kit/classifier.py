
import numpy as np
import scipy.ndimage as nd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier


class Classifier(BaseEstimator):
    def __init__(self):
        #self.clf = RandomForestClassifier(n_estimators=3, max_depth=10)
        self.clf = DummyClassifier(strategy="constant", constant=0)
        #self.shift = 2 # how many neighbours are taken for calculating features

    def _get_features(self, X):
        # very slow
        n_img, n_h, n_w, n_d = X.shape
        n_features = 2
        x_features = np.zeros((n_img, n_h, n_w, n_d, n_features))
        for img_no in range(n_img):
            for h in range(self.shift,n_h-self.shift):
                for w in range(self.shift,n_w-self.shift):
                    for d in range(self.shift,n_d-self.shift):
                        # calculate features: mean and std of all neighbouring voxels
                        cube = X[img_no,h-self.shift:h+self.shift,w-self.shift:w+self.shift,d-self.shift:d+self.shift]

                        x_features[img_no,h,w,d,:] = (cube.mean(), np.std(cube))
        x_features = x_features.reshape(n_img * n_h * n_w * n_d, n_features)
     
        return x_features
        
    def _get_features_scipy(self, X):
        
        n_features = 1
        n_img, n_h, n_w, n_d = X.shape
        
       
        x_features = np.zeros((n_img, n_h, n_w, n_d, n_features))
        # for every patient separately
        for img_no in range(n_img):
            # calculate gaussian filter
            x_features[img_no,:,:,:,0] = nd.gaussian_filter(X[img_no,:,:,:],sigma=5)
        
            # calculate uniform filter
            #x_features[img_no,:,:,:,1] = nd.uniform_filter(X[img_no,:,:,:])
        
        x_features = x_features.reshape(n_img * n_h * n_w * n_d, n_features)
        return x_features
        
    def _get_features_strided(self, X):

        # cut the borders
        n_img, n_h, n_w, n_d = X.shape
        n_features = 2
        new_shape = (n_img, n_h - (2 * self.shift), n_w - (2 * self.shift), n_d - (2 * self.shift),
                     (2 * self.shift), 2 * self.shift, 2 * self.shift)
        new_strides = X.strides + X.strides[1:]
        X_cube = np.lib.stride_tricks.as_strided(X, shape=new_shape, strides=new_strides, writeable = False)
        
        #  FIXME: produces nans
        #  calculate feature: STD avoiding creation of large temporary arrays
        Xsq_cube = np.lib.stride_tricks.as_strided(X * X, new_shape, new_strides)


        # calculate feature: means
        X_means = np.mean(X_cube, axis=(4, 5, 6))
        Xsq_means = np.mean(Xsq_cube, axis=(4, 5, 6))       
        X_stds = np.sqrt(Xsq_means - X_means * X_means)  
        
        # get voxel value
        #voxel = np.reshape(X, new_shape[:4])
        
        X_features = np.stack([X_means, voxel], axis=4).reshape(n_img * (n_h - 2 * self.shift) * (n_w - 2 * self.shift)  * (n_d - 2 * self.shift), n_features)
        
        return X_features      
          
    def _unpack_y(self, y, cut_bordering_pixels=False):
        # return y flattened and sparse matrices unpacked
        n_img, n_h = y.shape
        n_w, n_d = y[0][0].shape
        
        if cut_bordering_pixels:
            # cut out the bordering voxels (if you want to use it with strided function)     
            new_h = n_h - (2 * self.shift)
            new_w = n_w - (2 * self.shift)
            new_d = n_d - (2 * self.shift)
            y_new = np.zeros((n_img,new_h, new_w, new_d)) 
            
            for img_no in range(n_img):
                for h in range(new_h):
                    y_new[img_no, h, :, :] = y[img_no, h].todense()[self.shift:-self.shift,self.shift:-self.shift] 
                    y_new = y_new.reshape(n_img * new_h * new_w * new_d)
        else:
            y_new = np.zeros((n_img,n_h, n_w, n_d)) 
            
            for img_no in range(n_img):
                for h in range(n_h):
                    y_new[img_no, h, :, :] = y[img_no, h].todense()   
            y_new = y_new.reshape(n_img * n_h * n_w * n_d)     
        return y_new            
        
    def fit(self, X, y):
        # takes image ravels it and returns 1s where there are maxs
        return self

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



    def predict_proba(self, X):
        # X_features = self._get_features_scipy(X)
        X = X.ravel()[:, np.newaxis]
        y_pred = self.clf.predict_proba(X)
        print(y_pred)
        
        return np.hstack([y_pred, 1 - y_pred])
