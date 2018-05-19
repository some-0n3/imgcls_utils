"""This class contains code for the MNIST data set.

The MNIST is a data set of 70,000 monochrome 28x28 images of handwritten
digits.

For more information please see the `web site
<http://yann.lecun.com/exdb/mnist/>`_.
"""
import gzip
from os import makedirs
from os.path import join

import numpy

from . import DataSet, download


__all__ = ('MNIST', )


def load_data_file(path):
    """Load a data file."""
    with gzip.open(path) as fobj:
        data = numpy.frombuffer(fobj.read(), dtype='ubyte', offset=16)
        return data.reshape((-1, 1, 28, 28))


def load_label_file(path):
    """Load a label file."""
    with gzip.open(path) as fobj:
        labels = numpy.frombuffer(fobj.read(), dtype='ubyte', offset=8)
        return labels


def download_file(name, root, overwrite):
    """Download one of the files."""
    download(f'http://yann.lecun.com/exdb/mnist/{name}',
             join(root, name), overwrite)


class MNIST(DataSet):
    """The MNIST data set.

    The MNIST data set consist of 70,000 28x28 monochrome images from
    NIST. Each image shows a digit from 0 to 9. The training set
    contains 60,000, the validation set 10,000 samples.

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
    <http://yann.lecun.com/exdb/mnist/>`_.
    """

    cache_file = 'mnist.npz'

    def download(self, root='./_datasets', overwrite=False):
        root = join(root, 'mnist')
        makedirs(root, exist_ok=True)
        download_file('train-images-idx3-ubyte.gz', root, overwrite)
        download_file('train-labels-idx1-ubyte.gz', root, overwrite)
        download_file('t10k-images-idx3-ubyte.gz', root, overwrite)
        download_file('t10k-labels-idx1-ubyte.gz', root, overwrite)

    def extract(self, root='./_datasets'):
        return {
            'training data': load_data_file(
                join(root, 'mnist', 'train-images-idx3-ubyte.gz')),
            'training labels': load_label_file(
                join(root, 'mnist', 'train-labels-idx1-ubyte.gz')),
            'validation data': load_data_file(
                join(root, 'mnist', 't10k-images-idx3-ubyte.gz')),
            'validation labels': load_label_file(
                join(root, 'mnist', 't10k-labels-idx1-ubyte.gz'))
        }
