import pytest
from nilearn.image import load_img
import numpy as np

import estimator as estimator

@pytest.fixture
def init_est():
    image_size = (192, 224, 176)
    patch_shape = (192, 224, 8)
    epochs = 150
    batch_size = 1
    initial_learning_rate = 0.01
    learning_rate_drop = 0.5
    learning_rate_patience = 5
    early_stopping_patience = 10
    workers = 1
    est = estimator.KerasSegmentationClassifier(
        image_size, epochs=epochs, batch_size=batch_size,
        initial_learning_rate=initial_learning_rate,
        learning_rate_drop=learning_rate_drop,
        learning_rate_patience=learning_rate_patience,
        early_stopping_patience=early_stopping_patience,
        workers=workers, patch_shape=patch_shape
        )
    return est


@pytest.fixture
def data():
    """ returns 2 samples of X and y """
    X1_path = '././data/train/1_T1.nii.gz'
    X2_path = '././data/train/2_T1.nii.gz'

    y_path1 = '././data/train/1_lesion.nii.gz'
    y_path2 = '././data/train/2_lesion.nii.gz'

    y1 = load_img(y_path1).get_fdata()
    y = np.empty([2, y1.shape[0], y1.shape[1], y1.shape[2]])
    y[0, :] = y1
    y[1, :] = load_img(y_path2).get_fdata()

    return [X1_path, X2_path], y


def test_image_loader(data):
    """ check if the ImageLoader is returning expected data """
    X, y = data
    img_loader = estimator.ImageLoader(X, y)
    X1, y1 = img_loader.load(img_index=0)
    X2, y2 = img_loader.load(img_index=1)

    assert not np.all(X1 == X2)
    assert not np.all(y1 == y2)
    assert np.all(load_img(X[0]).get_fdata() == X1)
    assert np.all(load_img(X[1]).get_fdata() == X2)
    assert np.all(y[0, :] == y1)
    assert np.all(y[1, :] == y2)


def test_generator_correct_data_shape(init_est, data):
    """ """
    X, y = data
    init_est.patch_shape = None
    img_loader = estimator.ImageLoader(X, y)
    init_est._build_generator()
    import pdb; pdb.set_trace()


def test_n_steps_correct():
    pass


def test_skip_blank():
    pass


def test_generator_leads_to_new_data():
    pass


def test_simple_model():
    pass


def test_simple_deep():
    pass


def test_unet():
    pass
