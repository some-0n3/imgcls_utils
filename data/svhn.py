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
from scipy.io import loadmat

from . import DataSet, download


__all__ = ('SVHN', 'FullSVHN')


def load_file(filepath):
    """Load data and labels from a file."""
    read = loadmat(filepath)
    return read['X'].transpose(3, 2, 0, 1), read['y'].flatten() - 1


def download_file(name, root, overwrite=False):
    """Download one of the files."""
    download(f'http://ufldl.stanford.edu/housenumbers/{name}',
             join(root, name), overwrite)


class SVHN(DataSet):
    """The (small) Street View House Numbers data set.

    Street View House Numbers data set consist of 32x32 color images
    depicting the numbers from 1 to 10. Each image shows a number in
    it's center and may also a distracting number at either side.

    The data set consists of 73,257 training and 26,032 validation
    images.

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
    root: string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite: boolean (``False``)
        If ``True`` any existing data will be overwritten.

    For more information please see the `web site
    <http://ufldl.stanford.edu/housenumbers/>`_.
    """

    cache_file = 'svhn.npz'

    def download(self, root='./_datasets', overwrite=False):
        root = join(root, 'svhn')
        makedirs(root, exist_ok=True)
        download_file('train_32x32.mat', root, overwrite)
        download_file('test_32x32.mat', root, overwrite)

    def extract(self, root='./_datasets'):
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
    depicting the numbers from 1 to 10. Each image shows a number in
    it's center and may also a distracting number at either sides.

    This class provides a joined training set with 604,388 images and the
    validation set with 26,032 images.

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
    root: string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite: boolean (``False``)
        If ``True`` any existing data will be overwritten.

    For more information please see the `web site
    <http://ufldl.stanford.edu/housenumbers/>`_.
    """

    cache_file = 'svhn_full.npz'

    def download(self, root='./_datasets', overwrite=False):
        super(FullSVHN, self).download(root=root, overwrite=overwrite)
        download_file('extra_32x32.mat', join(root, 'svhn'), overwrite)

    def extract(self, root='./_datasets'):
        dct = super(FullSVHN, self).extract(root=root)
        ex_data, ex_labels = load_file(join(root, 'svhn', 'extra_32x32.mat'))
        tr_data, tr_labels = dct['training data'], dct['training labels']
        dct['original training data'] = tr_data
        dct['original training labels'] = tr_labels
        dct['extra data'] = ex_data
        dct['extra labels'] = ex_labels
        dct['training data'] = numpy.concatenate((tr_data, ex_data))
        dct['training labels'] = numpy.concatenate((tr_labels, ex_labels))
        return dct
