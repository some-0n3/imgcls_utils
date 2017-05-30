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
    download('http://yann.lecun.com/exdb/mnist/{}'.format(name),
             join(root, name), overwrite)


class MNIST(DataSet):
    """The MNIST data set.

    The MNIST data set consist of 70,000 28x28 monochrome images from
    NIST. Each image shows a digit from 0 to 9. The training set
    contains 60,000, the validation set 10,000 samples.

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
    <http://yann.lecun.com/exdb/mnist/>`_.
    """

    cache_file = 'mnist.pkl'

    def download(self, root='./_datasets', overwrite=False):
        root = join(root, 'mnist')
        makedirs(root, exist_ok=True)
        download_file('train-images-idx3-ubyte.gz', root, overwrite)
        download_file('train-labels-idx1-ubyte.gz', root, overwrite)
        download_file('t10k-images-idx3-ubyte.gz', root, overwrite)
        download_file('t10k-labels-idx1-ubyte.gz', root, overwrite)

    def create(self, root='./_datasets'):
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
