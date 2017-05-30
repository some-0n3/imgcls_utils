"""This module contains code for the CIFAR-10 and CIFAR-100 data sets.

Both consist of 32x32 color images, 50000 in the training set, and
10000 images in the test set.

The data and more information can be found at `Alex Krizhevsky's
homepage <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
"""
import pickle
import tarfile
from os.path import join

import numpy

from . import DataSet, split, download


__all__ = ('CIFAR10', 'CIFAR100')


def format_data(data, *labels):
    """Bring data and labels into ``numpy.array``s."""
    data = numpy.vstack(data).reshape(-1, 3, 32, 32)
    labels = [numpy.hstack(l) for l in labels]
    return data, labels


class CIFAR10(DataSet):
    """The CIFAR-10 data set.

    The CIFAR-10 data set contains 60,000 32x32 color images. The
    training set consist of 50,000 images the test set 10,000.
    The are 10 classes with overall 6,000 images per class. The
    classes have the labels: 'airplane', 'automobile', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship' and 'truck'.

    Parameters
    ----------
    testsplit : float in [0, 1] or integer (``0``)
        Says how may data points from the training set are reserved
        for the test set. If ``testsplit`` is ``0`` (or less) the
        validation set is used as test set.
        The parameter is either a float in [0, 1] describing the split
        in percent or an integer describing the number of elements.
    interval : tuple (pair) of numbers or ``None`` (``None``)
        The interval to put the data points into.
    root : string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite : boolean (``False``)
        If ``True`` any existing data will be overwritten.

    For more information please see the `web site
    <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
    """

    cache_file = 'cifar10.pkl'
    tar_file = 'cifar-10-python.tar.gz'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    def download(self, root='./_datasets', overwrite=False):
        download(self.url, join(root, self.tar_file), overwrite=overwrite)

    def create(self, root='./_datasets'):
        training = []
        result = {}
        with tarfile.open(join(root, self.tar_file)) as archive:
            # training data
            for bpath in ('cifar-10-batches-py/data_batch_{}'.format(i)
                          for i in range(1, 6)):
                member = archive.getmember(bpath)
                dct = pickle.load(archive.extractfile(member),
                                  encoding='bytes')
                training.append((dct[b'data'], dct[b'labels']))

            data = format_data(*zip(*training))
            result['training data'], (result['training labels'], ) = data

            # test data
            member = archive.getmember('cifar-10-batches-py/test_batch')
            dct = pickle.load(archive.extractfile(member), encoding='bytes')
            data = format_data((dct[b'data'], ), (dct[b'labels'], ))
            result['validation data'], (result['validation labels'], ) = data
        return result


class CIFAR100(DataSet):
    """The CIFAR-100 data set.

    As the CIFAR-10, the CIFAR-100 data set contains 60,000 32x32 color
    images. The training set consist of 50,000 images the test set
    10,000. The difference is that there are 100 classes with 600
    images per class.
    The 100 classes can be grouped into 20 super classes with 5
    sub classes each. The super classes are called coarse labels, the
    sub-classes are called to as fine labels.
    The label classes and super classes are:

    | super class                    | classes                         |
    |--------------------------------+---------------------------------|
    | aquatic mammals                | beaver, dolphin, otter, seal,   |
    |                                | whale                           |
    | fish                           | aquarium fish, flatfish, ray,   |
    |                                | shark, trout                    |
    | flowers                        | orchids, poppies, roses,        |
    |                                | sunflowers, tulips              |
    | food containers                | bottles, bowls, cans, cups,     |
    |                                | plates                          |
    | fruit and vegetables           | apples, mushrooms, oranges,     |
    |                                | pears, sweet peppers            |
    | household electrical devices   | clock, computer keyboard, lamp, |
    |                                | telephone, television           |
    | household furniture            | bed, chair, couch, table,       |
    |                                | wardrobe                        |
    | insects                        | bee, beetle, butterfly,         |
    |                                | caterpillar, cockroach          |
    | large carnivores               | bear, leopard, lion, tiger, wolf|
    | large man-made outdoor things  | bridge, castle, house, road,    |
    |                                | skyscraper                      |
    | large natural outdoor scenes   | cloud, forest, mountain, plain, |
    |                                | sea                             |
    | large omnivores and herbivores | camel, cattle, chimpanzee,      |
    |                                | elephant, kangaroo              |
    | medium-sized mammals           | fox, porcupine, possum, raccoon,|
    |                                | skunk                           |
    | non-insect invertebrates       | crab, lobster, snail, spider,   |
    |                                | worm                            |
    | people                         | baby, boy, girl, man, woman     |
    | reptiles                       | crocodile, dinosaur, lizard,    |
    |                                | snake, turtle                   |
    | small mammals                  | hamster, mouse, rabbit, shrew,  |
    |                                | squirrel                        |
    | trees                          | maple, oak, palm, pine, willow  |
    | vehicles 1                     | bicycle, bus, motorcycle,       |
    |                                | pickup truck, train             |
    | vehicles 2                     | lawn-mower, rocket, streetcar,  |
    |                                | tank, tractor                   |

    Parameters
    ----------
    testsplit : float in [0, 1] or integer (``0``)
        Says how may data points from the training set are reserved
        for the test set. If ``testsplit`` is ``0`` (or less) the
        validation set is used as test set.
        The parameter is either a float in [0, 1] describing the split
        in percent or an integer describing the number of elements.
    interval : tuple (pair) of numbers or ``None`` (``None``)
        The interval to put the data points into.
    root : string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite : boolean (``False``)
        If ``True`` any existing data will be overwritten.

    For more information please see the `web site
    <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
    """

    cache_file = 'cifar100.pkl'
    tar_file = 'cifar-100-python.tar.gz'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

    def __init__(self, testsplit=0, **kwargs):
        super(CIFAR100, self).__init__(testsplit=0, **kwargs)
        self._y_valid_fine = self.__dct__['fine validation labels'].astype(
            numpy.int32)
        self._y_valid_coarse = self.__dct__['coarse validation labels'].astype(
            numpy.int32)
        self._y_train_fine = self.__dct__['fine training labels'].astype(
            numpy.int32)
        self._y_train_coarse = self.__dct__['coarse training labels'].astype(
            numpy.int32)

        if testsplit > 0:
            self._x_train, itrain, self._x_test, itest = split(
                self._x_train, numpy.arange(len(self._x_train)), testsplit)
            self._y_test = self._y_test_fine = self._y_train_fine[itest]
            self._y_test_coarse = self._y_train_coarse[itest]
            self._y_train = self._y_train_fine = self._y_train[itrain]
            self._y_train_coarse = self._y_train_coarse[itrain]

    def download(self, root='./_datasets', overwrite=False):
        download(self.url, join(root, self.tar_file), overwrite=overwrite)

    def create(self, root='./_datasets'):
        def get_data(archive, name):
            """Retrieve data from an archived file."""
            member = archive.getmember('cifar-100-python/{}'.format(name))
            dct = pickle.load(archive.extractfile(member), encoding='bytes')
            return format_data(dct[b'data'], dct[b'coarse_labels'],
                               dct[b'fine_labels'])

        result = {}
        with tarfile.open(join(root, self.tar_file)) as archive:
            data, (clabels, flabels) = get_data(archive, 'train')
            result['training data'] = data
            result['fine training labels'] = flabels
            result['training labels'] = flabels
            result['coarse training labels'] = clabels

            data, (clabels, flabels) = get_data(archive, 'test')
            result['validation data'] = data
            result['validation labels'] = flabels
            result['fine validation labels'] = flabels
            result['coarse validation labels'] = clabels
        return result

    @property
    def validation_labels_fine(self):
        """The fine labels of the validation set."""
        return self._y_valid_fine

    @property
    def validation_labels_coarse(self):
        """The coarse labels of the validation set."""
        return self._y_valid_coarse

    @property
    def test_labels_fine(self):
        """The fine labels of the test set."""
        return self._y_test_fine

    @property
    def test_labels_coarse(self):
        """The coarse labels of the test set."""
        return self._y_test_coarse

    @property
    def training_labels_fine(self):
        """The fine labels of the training set."""
        return self._y_train_fine

    @property
    def training_labels_coarse(self):
        """The coarse labels of the training set."""
        return self._y_train_coarse
