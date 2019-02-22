
import numpy as np
import scipy.ndimage as nd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier


class SegmentationClassifier(BaseEstimator):
    def __init__(self):
        #self.clf = RandomForestClassifier(n_estimators=3, max_depth=10)
        self.clf = DummyClassifier(strategy="constant", constant=0)
        #self.shift = 2 # how many neighbours are taken for calculating features
          
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
        #return self
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


