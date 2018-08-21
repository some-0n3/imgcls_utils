from tempfile import NamedTemporaryFile

import numpy
import pytest
import theano
from decorator import contextmanager
from lasagne.utils import floatX

from .data import DataSet, cifar, cifar_lee14, mnist, mnist_distractor, svhn


def identical_dataset(set1, set2, max_bs=500):
    lenght = len(set1.data)
    batchsize = next(i for i in range(max_bs, 0, -1) if lenght % i == 0)
    assert lenght == len(set2.data)
    for (x1, y1), (x2, y2) in zip(set1.iter(batchsize), set2.iter(batchsize)):
        assert numpy.allclose(x1, x2)
        assert numpy.all(y1 == y2)
    return True


@contextmanager
def from_state_file(dataset, frmt='npz', **kwargs):
    tmp = NamedTemporaryFile(suffix=f'.{frmt}')
    try:
        dataset.save_state(tmp.name, **kwargs)
        yield type(dataset).from_state(tmp.name)
    finally:
        tmp.close()


def check_data(data):
    assert hasattr(data, 'data')
    assert hasattr(data, 'labels')
    assert hasattr(data, 'set')
    assert data.set == (data.data, data.labels)
    return True


def check_dims(data, data_shape):
    length = data_shape[0]
    assert data.data.shape == data_shape
    assert data.labels.shape == (length, )
    assert len(data.data) == length
    assert len(data.labels) == length
    return True


def check_interval(data, lmin, lmax, soft=False):
    lmin, lmax = floatX(lmin), floatX(lmax)
    if soft:
        return numpy.min(data) >= lmin and numpy.max(data) <= lmax
    return numpy.min(data) == lmin and numpy.max(data) == lmax


class Simple():

    @staticmethod
    def identical_data(baseline, copy):
        assert identical_dataset(baseline.training, copy.training)
        assert identical_dataset(baseline.validation, copy.validation)
        return True

    def test_create(self):
        dataset = self.dataset_cls()
        assert dataset.training
        assert check_data(dataset.training)
        assert dataset.validation
        assert check_data(dataset.validation)
        assert dataset.test is None

    def test_dims(self):
        dataset = self.dataset_cls()
        assert check_dims(dataset.training, self.train_shape)
        assert check_dims(dataset.validation, self.valid_shape)

    def test_dtypes(self):
        dataset = self.dataset_cls()

        data = dataset.training.data[:]
        assert data.dtype == theano.config.floatX
        assert dataset.training.labels.dtype == numpy.dtype('int32')

        data = dataset.validation.data[:]
        assert data.dtype == theano.config.floatX
        assert dataset.validation.labels.dtype == numpy.dtype('int32')

    def test_state(self):
        dataset = self.dataset_cls()
        state = dataset.state
        assert state
        assert isinstance(state, dict)

        assert 'training' in state
        assert state['training']
        assert isinstance(state['training'], dict)

        assert 'validation' in state
        assert state['validation']
        assert isinstance(state['validation'], dict)

        assert 'test' not in state

    def test_from_state(self):
        dataset = self.dataset_cls()
        copy = self.dataset_cls.from_state(dataset.state)
        assert self.identical_data(dataset, copy)

    def test_from_state_iter(self):
        dataset = self.dataset_cls()
        for _ in dataset.training.iter(100):
            pass
        copy = self.dataset_cls.from_state(dataset.state)
        assert self.identical_data(dataset, copy)

    def test_from_state_file_numpy(self):
        dataset = self.dataset_cls()
        with from_state_file(dataset) as copy:
            assert self.identical_data(dataset, copy)

    def test_from_state_file_numpy_compressed(self):
        dataset = self.dataset_cls()
        with from_state_file(dataset, compression=True) as copy:
            assert self.identical_data(dataset, copy)

    def test_from_state_file_h5(self):
        dataset = self.dataset_cls()
        with from_state_file(dataset, frmt='h5') as copy:
            assert self.identical_data(dataset, copy)

    def test_from_state_file_h5_chunked(self):
        dataset = self.dataset_cls()
        with from_state_file(dataset, frmt='h5', chunks=True) as copy:
            assert self.identical_data(dataset, copy)

    def test_from_state_file_h5_gzip(self):
        dataset = self.dataset_cls()
        with from_state_file(dataset, frmt='h5', compression='gzip') as copy:
            assert self.identical_data(dataset, copy)

    def test_from_state_file_h5_lzf(self):
        dataset = self.dataset_cls()
        with from_state_file(dataset, frmt='h5', compression='lzf') as copy:
            assert self.identical_data(dataset, copy)

    def test_from_state_file_h5_szip(self):
        dataset = self.dataset_cls()
        with from_state_file(dataset, frmt='h5', compression='szip') as copy:
            assert self.identical_data(dataset, copy)

    def test_from_state_file_h5_gzip_0(self):
        dataset = self.dataset_cls()
        with from_state_file(dataset, frmt='h5', compression='gzip',
                             compression_opts=0) as copy:
            assert self.identical_data(dataset, copy)

    def test_from_state_file_h5_gzip_9(self):
        dataset = self.dataset_cls()
        with from_state_file(dataset, frmt='h5', compression='gzip',
                             compression_opts=9) as copy:
            assert self.identical_data(dataset, copy)

    def test_from_state_file_h5_shuffle(self):
        dataset = self.dataset_cls()
        with from_state_file(dataset, frmt='h5', shuffle=True) as copy:
            assert self.identical_data(dataset, copy)

    def test_from_state_file_h5_shuffle_lzf(self):
        dataset = self.dataset_cls()
        with from_state_file(dataset, frmt='h5', compression='lzf',
                             shuffle=True) as copy:
            assert self.identical_data(dataset, copy)

    def test_from_state_file_h5_fletcher32(self):
        dataset = self.dataset_cls()
        with from_state_file(dataset, frmt='h5', fletcher32=True) as copy:
            assert self.identical_data(dataset, copy)

    def test_from_state_file_h5_someconf(self):
        dataset = self.dataset_cls()
        with from_state_file(dataset, frmt='h5', compression='lzf',
                             fletcher32=True, shuffle=True) as copy:
            assert self.identical_data(dataset, copy)

    def test_all_labels(self):
        dataset = self.dataset_cls()
        labels = numpy.concatenate((dataset.training.labels,
                                    dataset.validation.labels))
        labels = set(labels)
        assert len(labels) == self.num_labels
        assert tuple(range(len(labels))) == tuple(sorted(labels))

    def test_train_labels(self):
        dataset = self.dataset_cls()
        labels = set(dataset.training.labels)
        assert len(labels) == self.num_labels
        assert tuple(range(len(labels))) == tuple(sorted(labels))

    def test_valid_labels(self):
        dataset = self.dataset_cls()
        labels = set(dataset.validation.labels)
        assert len(labels) == self.num_labels
        assert tuple(range(len(labels))) == tuple(sorted(labels))

    def test_interval(self):
        dataset = self.dataset_cls()
        assert check_interval(dataset.training.data, 0, 1)
        assert check_interval(dataset.validation.data, 0, 1)

        dataset = self.dataset_cls(interval=None)
        assert check_interval(dataset.training.data, *self.interval, soft=True)
        assert check_interval(dataset.validation.data, *self.interval,
                              soft=True)

        dataset = self.dataset_cls(interval=(-1, 1))
        assert check_interval(dataset.training.data, -1, 1)
        assert check_interval(dataset.validation.data, -1, 1)

        dataset = self.dataset_cls(interval=(-7, -1))
        assert check_interval(dataset.training.data, -7, -1)
        assert check_interval(dataset.validation.data, -7, -1)

        dataset = self.dataset_cls(interval=(3.5, 4.4))
        assert check_interval(dataset.training.data, 3.5, 4.4)
        assert check_interval(dataset.validation.data, 3.5, 4.4)

        dataset = self.dataset_cls(interval=(-3, 0))
        assert check_interval(dataset.training.data, -3, 0)
        assert check_interval(dataset.validation.data, -3, 0)

        with pytest.raises(ValueError):
            dataset = self.dataset_cls(interval=(2, -1))
        with pytest.raises(ValueError):
            dataset = self.dataset_cls(interval=(-6, -12))
        with pytest.raises(ValueError):
            dataset = self.dataset_cls(interval=(0, 0))
        with pytest.raises(ValueError):
            dataset = self.dataset_cls(interval=(1, 1))


class SplitNumber(Simple):

    def test_create(self):
        super(SplitNumber, self).test_create()

        dataset = self.dataset_cls(testsplit=100)
        assert dataset.test

        dataset = self.dataset_cls(testsplit=None)
        assert dataset.test is None

        dataset = self.dataset_cls(testsplit=0)
        assert dataset.test is None

        dataset = self.dataset_cls(testsplit=0.2)
        assert dataset.test

        with pytest.raises(ValueError):
            dataset = self.dataset_cls(testsplit=-0.4)
        with pytest.raises(ValueError):
            dataset = self.dataset_cls(testsplit=-200)
        with pytest.raises(ValueError):
            dataset = self.dataset_cls(testsplit=1000000000)

    def test_dims(self):
        super(SplitNumber, self).test_dims()

        dataset = self.dataset_cls(testsplit=100)
        train_shape = (self.train_shape[0] - 100, ) + self.train_shape[1:]
        test_shape = (100, ) + self.train_shape[1:]
        assert check_dims(dataset.training, train_shape)
        assert check_dims(dataset.test, test_shape)

        dataset = self.dataset_cls(testsplit=0.2)
        split = int(0.2 * self.train_shape[0])
        train_shape = (self.train_shape[0] - split, ) + self.train_shape[1:]
        test_shape = (split, ) + self.train_shape[1:]
        assert check_dims(dataset.training, train_shape)
        assert check_dims(dataset.test, test_shape)

    def test_dtypes(self):
        super(SplitNumber, self).test_dtypes()
        dataset = self.dataset_cls(testsplit=100)
        data = dataset.test.data[:]
        assert data.dtype == theano.config.floatX
        assert dataset.test.labels.dtype == numpy.dtype('int32')

    def test_state(self):
        super(SplitNumber, self).test_state()

        dataset = self.dataset_cls(testsplit=100)
        assert dataset.state['test']
        assert isinstance(dataset.state['test'], dict)

        dataset = self.dataset_cls(testsplit=None)
        assert 'test' not in dataset.state

        dataset = self.dataset_cls(testsplit=0)
        assert 'test' not in dataset.state

        dataset = self.dataset_cls(testsplit=0.2)
        assert dataset.state['test']
        assert isinstance(dataset.state['test'], dict)

    def test_from_state(self):
        super(SplitNumber, self).test_from_state()
        dataset = self.dataset_cls(testsplit=100)
        copy = self.dataset_cls.from_state(dataset.state)
        assert identical_dataset(dataset.training, copy.training)
        assert identical_dataset(dataset.test, copy.test)

    def test_from_state_file_numpy(self):
        super(SplitNumber, self).test_from_state_file_numpy()
        dataset = self.dataset_cls(testsplit=100)
        with from_state_file(dataset) as copy:
            assert identical_dataset(dataset.training, copy.training)
            assert identical_dataset(dataset.test, copy.test)

    def test_from_state_file_h5(self):
        super(SplitNumber, self).test_from_state_file_h5()
        dataset = self.dataset_cls(testsplit=100)
        with from_state_file(dataset, frmt='h5') as copy:
            assert identical_dataset(dataset.training, copy.training)
            assert identical_dataset(dataset.test, copy.test)

    def test_interval(self):
        super(SplitNumber, self).test_interval()

        dataset = self.dataset_cls(testsplit=100)
        assert check_interval(dataset.training.data, 0, 1, soft=True)
        assert check_interval(dataset.test.data, 0, 1, soft=True)

        dataset = self.dataset_cls(interval=None, testsplit=0.1)
        assert check_interval(dataset.training.data, *self.interval, soft=True)
        assert check_interval(dataset.test.data, *self.interval, soft=True)

        dataset = self.dataset_cls(interval=(-1, 1), testsplit=0.1)
        assert check_interval(dataset.training.data, -1, 1, soft=True)
        assert check_interval(dataset.test.data, -1, 1, soft=True)

        dataset = self.dataset_cls(interval=(-7, -1), testsplit=150)
        assert check_interval(dataset.training.data, -7, -1, soft=True)
        assert check_interval(dataset.test.data, -7, -1, soft=True)

        dataset = self.dataset_cls(interval=(3.5, 4.4), testsplit=200)
        assert check_interval(dataset.training.data, 3.5, 4.4, soft=True)
        assert check_interval(dataset.test.data, 3.5, 4.4, soft=True)

        dataset = self.dataset_cls(interval=(-3, 0), testsplit=0.2)
        assert check_interval(dataset.training.data, -3, 0, soft=True)
        assert check_interval(dataset.test.data, -3, 0, soft=True)


class SplitValid(Simple):

    def test_create(self):
        super(SplitValid, self).test_create()
        dataset = self.dataset_cls(testsplit='validation')
        assert dataset.test
        assert dataset.validation
        assert dataset.test is dataset.validation

    def test_state(self):
        super(SplitValid, self).test_state()
        dataset = self.dataset_cls(testsplit='validation')
        state = dataset.state
        assert state['test'] == 'validation'

    def test_from_state(self):
        super(SplitValid, self).test_from_state()
        dataset = self.dataset_cls(testsplit='validation')
        copy = self.dataset_cls.from_state(dataset.state)
        assert copy.test is copy.validation
        assert identical_dataset(dataset.test, copy.test)

    def test_from_state_file_numpy(self):
        super(SplitValid, self).test_from_state_file_numpy()
        dataset = self.dataset_cls(testsplit='validation')
        with from_state_file(dataset) as copy:
            assert copy.test is copy.validation
            assert identical_dataset(dataset.test, copy.test)

    def test_from_state_file_h5(self):
        super(SplitValid, self).test_from_state_file_h5()
        dataset = self.dataset_cls(testsplit='validation')
        with from_state_file(dataset, frmt='h5') as copy:
            assert copy.test is copy.validation
            assert identical_dataset(dataset.test, copy.test)


class Base(SplitNumber, SplitValid, Simple):
    pass


def check_dict(dct):
    assert isinstance(dct, dict)

    def check(prefix):
        x_key = f'{prefix} data'
        y_key = f'{prefix} labels'
        assert x_key in dct
        assert y_key in dct

        x_val = dct[x_key]
        y_val = dct[y_key]
        assert x_val.shape[0] == y_val.shape[0]
        assert len(x_val) == len(y_val)

        return True

    assert check('training')
    assert check('validation')
    return True


class RealMixin():

    def test_download(self):
        pytest.skip()

    def test_extract(self):
        dataset = self.dataset_cls()
        assert check_dict(dataset.extract())

    def test_load(self):
        dataset = self.dataset_cls()
        assert check_dict(dataset.load())


class Real(RealMixin, Base):
    pass


class ExampleDataSet(DataSet):
    """Example data set for testing."""

    def download(*args, **kwargs):
        pass

    def extract(self, *args, **kwargs):
        return {
            'training data': numpy.random.uniform(0, 256, (1000, 1, 16, 16)),
            'training labels': numpy.random.uniform(0, 10, 1000),
            'validation data': numpy.random.uniform(0, 256, (200, 1, 16, 16)),
            'validation labels': numpy.random.uniform(0, 10, 200)
        }


class TestExample(Base):

    dataset_cls = ExampleDataSet
    train_shape = (1000, 1, 16, 16)
    valid_shape = (200, 1, 16, 16)
    interval = (0, 256)
    num_labels = 10


class TestCIFAR10(Real):

    dataset_cls = cifar.CIFAR10
    train_shape = (50000, 3, 32, 32)
    valid_shape = (10000, 3, 32, 32)
    interval = (0, 255)
    num_labels = 10


class TestCIFAR100(Real):

    dataset_cls = cifar.CIFAR100
    train_shape = (50000, 3, 32, 32)
    valid_shape = (10000, 3, 32, 32)
    interval = (0, 255)
    num_labels = 100


class TestMNIST(Real):

    dataset_cls = mnist.MNIST
    train_shape = (60000, 1, 28, 28)
    valid_shape = (10000, 1, 28, 28)
    interval = (0, 255)
    num_labels = 10


class TestSVHN(Real):

    dataset_cls = svhn.SVHN
    train_shape = (73257, 3, 32, 32)
    valid_shape = (26032, 3, 32, 32)
    interval = (0, 255)
    num_labels = 10


class TestFullSVHN(RealMixin, Simple):

    dataset_cls = svhn.FullSVHN
    train_shape = (604388, 3, 32, 32)
    valid_shape = (26032, 3, 32, 32)
    interval = (0, 255)
    num_labels = 10

    def test_interval(self):
        pytest.skip()

    def test_from_state_file_numpy(self):
        pytest.skip()

    def test_from_state_file_numpy_compressed(self):
        pytest.skip()


class TestMNISTDistractor(Real):

    dataset_cls = mnist_distractor.Distractor
    train_shape = (60000, 1, 28, 56)
    valid_shape = (10000, 1, 28, 56)
    interval = (0, 255)
    num_labels = 2


class TestAugCIFAR10(Real):

    dataset_cls = cifar_lee14.CIFAR10
    train_shape = (50000, 3, 32, 32)
    valid_shape = (10000, 3, 32, 32)
    num_labels = 10

    def test_interval(self):
        with pytest.raises(ValueError):
            self.dataset_cls(interval=(0, 1))


class TestAugCIFAR100(Real):

    dataset_cls = cifar_lee14.CIFAR100
    train_shape = (50000, 3, 32, 32)
    valid_shape = (10000, 3, 32, 32)
    num_labels = 100

    def test_interval(self):
        with pytest.raises(ValueError):
            self.dataset_cls(interval=(0, 1))
