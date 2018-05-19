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

from . import DataSet, download


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
    testsplit : ``None``, ``'validation'`` or positive number (``None``)
        Create a hold out test set (or not) from some of the training
        data. In case of ``None``, no test set will be created. The
        string ``'validation'`` will use the validation set for this.
        In case ``testsplit`` is a number the test set will be randomly
        drawn from the training set. If the number is an integer it will
        specify the number of examples in the test set. A float in [0, 1]
        describes the split in percent.
    interval : tuple (pair) of numbers or ``None`` (``(0, 1)``)
        The interval to put the data points into. In case of ``None`` the
        interval will not be changed.
    root : string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite : boolean (``False``)
        If ``True`` any existing data will be overwritten.

    For more information please see the `web site
    <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
    """

    cache_file = 'cifar10.npz'
    tar_file = 'cifar-10-python.tar.gz'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    def download(self, root='./_datasets', overwrite=False):
        download(self.url, join(root, self.tar_file), overwrite=overwrite)

    def extract(self, root='./_datasets'):
        training = []
        result = {}
        with tarfile.open(join(root, self.tar_file)) as archive:
            # training data
            for bpath in (f'cifar-10-batches-py/data_batch_{i}'
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
    testsplit : ``None``, ``'validation'`` or positive number (``None``)
        Create a hold out test set (or not) from some of the training
        data. In case of ``None``, no test set will be created. The
        string ``'validation'`` will use the validation set for this.
        In case ``testsplit`` is a number the test set will be randomly
        drawn from the training set. If the number is an integer it will
        specify the number of examples in the test set. A float in [0, 1]
        describes the split in percent.
    interval : tuple (pair) of numbers or ``None`` (``(0, 1)``)
        The interval to put the data points into. In case of ``None`` the
        interval will not be changed.
    root : string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite : boolean (``False``)
        If ``True`` any existing data will be overwritten.

    For more information please see the `web site
    <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
    """

    cache_file = 'cifar100.npz'
    tar_file = 'cifar-100-python.tar.gz'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

    def download(self, root='./_datasets', overwrite=False):
        download(self.url, join(root, self.tar_file), overwrite=overwrite)

    def extract(self, root='./_datasets'):
        def get_data(archive, name):
            """Retrieve data from an archived file."""
            member = archive.getmember(f'cifar-100-python/{name}')
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
