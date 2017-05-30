"""This module provides some data set driven utilities.

The module provides some functions for creating/downloading a benchmark
data set. Those data sets usually consist of a training and a validation
set. They also provide a method to create a test set by taking samples
from the training set.

The classes (by default) download all necessary files into a given
directory, they also write caching files into there.
"""
import pickle
from os import makedirs
from os.path import exists, join
from subprocess import run

import numpy
from lasagne.utils import floatX


__all__ = ('batch_iter', 'change_interval', 'download', 'split', 'DataSet')


def download(url, path, overwrite=False):
    """Download a URL and save it into a file.

    Parameters
    ----------
    url : string
        The URL to download.
    path : string
        The destination file path.
    overwrite : boolean (``False``)
        Existing files will be overwritten if ``True``.
    """
    # TODO : find a better way to download files
    if exists(path) and not overwrite:
        return
    try:
        run(['wget', url, '-O', path], check=True)
    except OSError:
        run(['curl', '-o', path, url], check=True)


def change_interval(data, old=(0, 255), new=(0.0, 1.0)):
    """Move a matrix from one interval into another.

    Parameters
    ----------
    data : ``numpy.array``
       The data points.
    old : tuple (pair) of numbers (``(0, 255)``)
        The old interval for the data.
    new : tuple (pair) of numbers (``(0.0, 1.0)``)
        The new interval for the data.

    Returns
    -------
    ``numpy.array``
        The data points in the new interval.
    """
    # TODO : make old optional, take min/max if not present
    return (data - old[0]) / (old[1] - old[0]) * (new[1] - new[0]) + new[0]


def split(data, labels, split):
    """Shuffle and split data.

    Parameters
    ----------
    data : list or ``numpy.array``
        The data points.
    labels : list or ``numpy.array``
        The label information.
    split : float in [0, 1] or integer
        Either a float in [0, 1] to describe the split in percent
        or an integer to describe the number of test cases.

    Returns
    -------
    ``numpy.array``
        The training data.
    ``numpy.array``
        The training labels.
    ``numpy.array``
        The test data.
    ``numpy.array``
        The test labels.
    """
    length = len(data)
    if split < 1:
        split = int(split * length)
    idx = numpy.random.permutation(length)
    idx_train = idx[:-split]
    idx_test = idx[-split:]
    return data[idx_train], labels[idx_train], data[idx_test], labels[idx_test]


def batch_iter(data, labels, batchsize, indices=None, shuffle=False):
    """Iterate over mini-batches of the data.

    Parameters
    ----------
    data : ``numpy.array``
        The data points.
    labels : ``numpy.array``
        The label information.
    batchsize : integer
        The number of data point per batch.
    indices : list or ``numpy.array`` of integers or ``None`` (``None``)
        Only use elements with these indices, if ``None`` all elements
        are used.
    shuffle : boolean (``False``)
        If ``True`` the data is shuffled before iterating over it.

    Yields
    ------
    ``numpy.array``
        The data points of the current batch.
    ``numpy.array``
        The label information for the current batch.
    """
    if indices is None:
        indices = numpy.arange(len(data))
    if shuffle:
        numpy.random.shuffle(indices)
    for idx in range(0, len(indices) - batchsize + 1, batchsize):
        batch = indices[idx:idx+batchsize]
        yield data[batch], labels[batch]


class DataSet(object):
    """The basic super class for all data sets.

    A basic :class:`DataSet` has a training, test and validation set.
    If there is no specified test set the validation set will be taken.
    The class downloads and processes all required files autonomously
    and caches them in a given directory.

    Parameters
    ----------
    testsplit : float in [0, 1] or integer (``0``)
        Says how may data points from the training set are reserved
        for the test set, if ``testsplit`` is ``0`` (or less) the
        validation set is used as test set.
        The parameter is either a float in [0, 1] describing the split
        in percent or an integer describing the number of elements.
    interval : tuple (pair) of numbers or ``None`` (``None``)
        The interval to put the data points into.
    root : string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite : boolean (``False``)
        If ``True`` any existing data will be overwritten.
    """

    def __init__(self, testsplit=0, interval=None,
                 root='./_datasets', overwrite=False):
        makedirs(root, exist_ok=True)
        if hasattr(self, 'cache_file'):
            path = join(root, self.cache_file)
            if not exists(path) or overwrite:
                self.download(root=root)
                dct = self.create(root=root)
                with open(path, 'wb') as fobj:
                    pickle.dump(dct, fobj)
            else:
                with open(path, 'rb') as fobj:
                    dct = pickle.load(fobj)
        else:
            self.download(root=root)
            dct = self.create(root=root)

        self.__dct__ = dct
        data = floatX(dct['training data'])
        self._x_valid = floatX(dct['validation data'])
        labels = dct['training labels'].astype(numpy.int32)
        self._y_valid = dct['validation labels'].astype(numpy.int32)

        if interval is not None:
            data = change_interval(data, new=interval)
            self._x_valid = change_interval(self._x_valid, new=interval)

        if testsplit > 0:
            dtr, ltr, dte, lte = split(data, labels, testsplit)
            self._x_train = dtr
            self._y_train = ltr
            self._x_test = dte
            self._y_test = lte
        else:
            self._x_train = data
            self._y_train = labels
            self._x_test = self._x_valid
            self._y_test = self._y_valid

    @staticmethod
    def download(root='./_datasets', overwrite=False):
        """Download all files necessary for the data set.

        Parameters
        ----------
        root : string (``'./_datasets'``)
            A directory to download the files into.
        overwrite : boolean (``False``)
            If ``True`` any existing data will be overwritten.
        """
        raise NotImplementedError('')

    def create(self, root='./_datasets'):
        """Create the data set.

        Parameters
        ----------
        root : string (``'./_datasets'``)
            The root-directory for the downloaded files.

        Returns
        -------
        dictionary
            The data set.
        """
        raise NotImplementedError('')

    @property
    def validation_data(self):
        """The data of the validation set."""
        return self._x_valid

    @property
    def validation_labels(self):
        """The labels of the validation set."""
        return self._y_valid

    @property
    def validation_set(self):
        """The validation data and labels."""
        return self.validation_data, self.validation_labels

    @property
    def test_data(self):
        """The data of the test set."""
        return self._x_test

    @property
    def test_labels(self):
        """The labels of the test set."""
        return self._y_test

    @property
    def test_set(self):
        """The test data and labels."""
        return self.test_data, self.test_labels

    @property
    def training_data(self):
        """The data of the training set."""
        return self._x_train

    @property
    def training_labels(self):
        """The labels of the training set."""
        return self._y_train

    @property
    def training_set(self):
        """The training data and labels."""
        return self.training_data, self.training_labels

    def endless_training_iter(self, batchsize):
        """Iterate endlessly over the training data.

        Parameters
        ----------
        batchsize : integer
            The number of elements in a batch.

        Yields
        ------
        ``numpy.array``
            The data points of the current batch.
        ``numpy.array``
            The label information for the current batch.
        """
        indices = tuple(range(len(self.training_data)))
        todos = []
        data, labels = self.training_set
        while True:
            todos.extend(numpy.random.permutation(indices))
            length = (len(todos) // batchsize) * batchsize
            yield from batch_iter(data, labels, batchsize,
                                  indices=todos[:length], shuffle=False)
            todos = todos[length:]
