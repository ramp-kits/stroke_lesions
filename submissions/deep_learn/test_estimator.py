import pytest
from nilearn.image import load_img
import numpy as np

import estimator as estimator


def init_est(**params):
    est_params = {
        'image_size': (192, 224, 8),
        'epochs': 150,
        'batch_size': 1,
        'initial_learning_rate': 0.01,
        'learning_rate_drop': 0.5,
        'learning_rate_patience': 5,
        'early_stopping_patience': 10,
        'workers': 1
    }
    # overwrite with given values
    for key, value in params.items():
        est_params[key] = value

    est = estimator.KerasSegmentationClassifier(**est_params)
    return est


@pytest.fixture
def data():
    """ returns 2 samples of X and y """
    X1_path = '././data/train/1_T1.nii.gz'
    X2_path = '././data/train/2_T1.nii.gz'

    y_path1 = '././data/train/1_lesion.nii.gz'
    y_path2 = '././data/train/2_lesion.nii.gz'

    return [X1_path, X2_path], [y_path1, y_path2]


class TestImageLoader():

    def __init__(self, X, y=None):
        self.X = X
        self.n_paths = len(X)
        self.y = y

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def load(self, img_index):
        if self.y is not None:
            return self.X[img_index], self.y[img_index]
        else:
            return self.X[img_index]

    def load_y(self, img_index):
        assert self.y is not None
        return self.y[img_index]


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
    assert np.all(load_img(y[0]).get_fdata() == y1)
    assert np.all(load_img(y[1]).get_fdata() == y2)


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("indices", [None, [1, 0], [0, 1]])
def test_generator_correct_output(data, train, shuffle, indices):
    """ it checks if generator lead to the correct output if shuffle is set to
    False and there are no patches (full images)"""
    X, y = data
    image_size = (197, 233, 189)
    params = {
        'image_size': image_size,
        'patch_shape': None,
        'skip_blank': False,
        'depth': 1
    }
    est = init_est(**params)

    if train:
        img_loader = estimator.ImageLoader(X, y)
    else:
        img_loader = estimator.ImageLoader(X)

    generator = est._build_generator(img_loader, shuffle=shuffle,
                                     train=train, indices=indices)
    if train:
        x1, y1 = next(generator)
        assert np.all(x1.shape == y1.shape)
        assert len(np.unique(y1)) in [1, 2]  # TODO: if training on the whole
        # image or with skip_blank set to true it should always be 2
        if not shuffle and not indices:
            assert np.all(y1[0, 0, ...] == load_img(y[0]).get_fdata())
        y1 = y1.copy()
    else:
        x1 = next(generator)
    x1 = x1.copy()
    assert len(np.unique(x1)) > 2

    # copy is necessary, otherwise it will reference the same loc in the memory
    # for x1 and x2 etc
    if train:
        x2, y2 = next(generator)
        y2 = y2.copy()
        assert not np.all(y1 == y2)
        if not shuffle and not indices:
            assert np.all(y2[0, 0, ...] == load_img(y[1]).get_fdata())
    else:
        x2 = next(generator)
    x2 = x2.copy()
    assert not np.all(x1 == x2)
    assert len(np.unique(x2)) > 2

    if train:
        x3, y3 = next(generator)
        y3 = y3.copy()
        assert np.all(y1 == y3)
    else:
        # the generator should pass only once through the data
        with pytest.raises(StopIteration):
            x3 = next(generator)

    if train:
        assert np.all(x1 == x3)

        if train:
            x4, y4 = next(generator)
            assert np.all(y2 == y4)
        else:
            x4 = next(generator)
        assert np.all(x2 == x4)


def test_generator_with_patches(data):
    pass


@pytest.mark.parametrize("model_type", ['simple_unet', 'simple', 'unet'])
def test_model_runs(model_type):
    n_samples = 50
    x_len, y_len, z_len = (8, 8, 8)

    # initiate the estimator:
    params = {
        'image_loader_factory': TestImageLoader,
        'image_size': (x_len, y_len, z_len),
        'patch_shape': None,
        'epochs': 1,
        'model_type': model_type
    }
    est = init_est(**params)

    y = np.random.choice([0, 1], size=n_samples * x_len * y_len * z_len)
    y = y.reshape(n_samples, x_len, y_len, z_len)
    X = np.random.rand(n_samples, x_len, y_len, z_len)

    est.fit(X, y)
    y_pred = est.predict(X)
    assert y_pred.shape == X.shape
    assert len(np.unique(y_pred)) in [1, 2]


def test_generator_with_batches():
    pass


def test_correct_n_steps():
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

def test_ensure_dice_problem_same_dice_estimator(data):
    import sys
    import tensorflow.keras.backend as K
    sys.path.append('.')
    sys.path.append('data/train/')
    x, y = data
    
    # dice for the problem.py
    from problem import DiceCoeff
    
    y_true = load_img(y[0]).get_fdata().astype('int32')

    y_true_tensor = K.constant(y_true)
    score = estimator._dice_coefficient(y_true_tensor, y_true_tensor)
    estimator_dice = float(score)
    
    diceclass = DiceCoeff()
    zz = 'data/train/1_lesion.nii.gz'
    problem_dice = diceclass.__call__([zz], [y_true])
    assert problem_dice == estimator_dice
    
