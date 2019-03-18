import numpy as np
import scipy.ndimage as nd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from keras.layers import Input, MaxPooling3D, UpSampling3D, Conv3D, Reshape, Conv3DTranspose

import problem
import numpy as np
import matplotlib.pylab as plt
import submissions.starting_kit.keras_segmentation_classifier as classifier 

module_path = '.'
train_ids = problem.get_train_data() 
print(train_ids)

#spl = problem.ImageLoader([1,2,3,4])

simp = problem.SimplifiedSegmentationClassifier() 
clf = simp.train_submission(module_path=module_path,patient_ids=train_ids)  


# n_classes=[0,1]
# img_loader = problem.ImageLoader(patient_ids=train_ids, n_classes=n_classes) 
# clf.fit(img_loader)  

#test_ids = problem.get_test_data()
#score = simp.test_submission(module_path = module_path,trained_model = clf, patient_idxs = test_ids)  


'''
train = problem.get_train_data()  
X, y = train

pred = classifier.Classifier().predict(X)
pred_prob = classifier.Classifier().predict_proba(X)
fitit = classifier.Classifier().fit(X, y)



classifier.Classifier().predict(X)
features = classifier.Classifier()._get_features_strided(X) 
y_new = classifier.Classifier()._unpack_y(y)



split_ids = problem._read_ids('.') 
path = '.' 
subject_id = 31970

X = np.stack([problem._read_brain_image(path, subject_id) for subject_id in split_ids]) 
Y = np.stack([problem._read_stroke_segmentation(path, subject_id) for subject_id in split_ids]) 


sys.getsizeof(test) 


problem._get_pati

plt.subplot(1,2,1)
plt.title('data')
plt.imshow(train[0,:,:,100])


plt.subplot(1,2,2)
plt.title('mask')
 
