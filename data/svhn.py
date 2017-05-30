"""This module contains code for the Street View House Numbers data set.

The data set consist of 32x32 color images obtained from house numbers
in Google Street View images.
This module contains two classes. One only provides the 73,257 training
and the 26,032 validation images. The other one also provides 531,131
"somewhat less difficult" extra data.

More information can be found on the `web site
<http://ufldl.stanford.edu/housenumbers/>`_.

So far the module only supports the "Cropped Digits" version of the
image data.
"""
from os import makedirs
from os.path import join

import numpy
from lasagne.utils import floatX
from scipy.io import loadmat

from . import DataSet, change_interval, download


__all__ = ('SVHN', 'FullSVHN')


def load_file(filepath):
    """Load data and labels from a file."""
    read = loadmat(filepath)
    return read['X'].transpose(3, 2, 0, 1), read['y'].flatten()


def download_file(name, root, overwrite=False):
    """Download one of the files."""
    download('http://ufldl.stanford.edu/housenumbers/{}'.format(name),
             join(root, name), overwrite)


class SVHN(DataSet):
    """The (small) Street View House Numbers data set.

    Street View House Numbers data set consist of 32x32 color images
    depicting the digits from 0 to 9. Each image shows a digit in
    it's center and may also a distracting digit at either side.

    The data set consists of 73,257 training and 26,032 validation
    images.

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
    root: string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite: boolean (``False``)
        If ``True`` any existing data will be overwritten.

    For more information please see the `web site
    <http://ufldl.stanford.edu/housenumbers/>`_.
    """

    cache_file = 'svhn.pkl'

    def download(self, root='./_datasets', overwrite=False):
        root = join(root, 'svhn')
        makedirs(root, exist_ok=True)
        download_file('train_32x32.mat', root, overwrite)
        download_file('test_32x32.mat', root, overwrite)

    def create(self, root='./_datasets'):
        train_data, train_labels = load_file(
            join(root, 'svhn', 'train_32x32.mat'))
        valid_data, valid_labels = load_file(
            join(root, 'svhn', 'test_32x32.mat'))
        return {
            'training data': train_data, 'training labels': train_labels,
            'validation data': valid_data, 'validation labels': valid_labels
        }


class FullSVHN(SVHN):
    """The (full) Street View House Numbers data set.

    Street View House Numbers data set consist of 32x32 color images
    depicting the digits from 0 to 9. Each image shows a digit in
    it's center and may also a distracting digit at either side.

    This class provides the 73,257 training and 26,032 validation, as
    well as 531,131 extra samples.

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
    root: string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite: boolean (``False``)
        If ``True`` any existing data will be overwritten.

    For more information please see the `web site
    <http://ufldl.stanford.edu/housenumbers/>`_.
    """

    cache_file = 'svhn_full.pkl'

    def __init__(self, *args, interval=None, **kwargs):
        super(FullSVHN, self).__init__(*args, interval=interval, **kwargs)
        self._x_extra = floatX(self.__dct__['extra data'])
        if interval is not None:
            self._x_extra = change_interval(self._x_extra, new=interval)
        self._y_extra = self.__dct__['extra labels'].astype(numpy.int32)

    def download(self, root='./_datasets', overwrite=False):
        super(FullSVHN, self).download(root=root, overwrite=overwrite)
        download_file('extra_32x32.mat', join(root, 'svhn'), overwrite)

    def create(self, root='./_datasets'):
        dct = super(FullSVHN, self).create(root=root)
        data, labels = load_file(join(root, 'svhn', 'extra_32x32.mat'))
        dct['extra data'], dct['extra labels'] = data, labels
        return dct

    @property
    def extra_data(self):
        """The data from the extra samples."""
        return self._x_extra

    @property
    def extra_labels(self):
        """The labels from the extra samples."""
        return self._y_extra

    @property
    def extra_set(self):
        """The extra data and labels."""
        return self.extra_data, self.extra_labels
