"""A Data set sampled from the full data set SVHN.

This module contains a data set that is sampled from the SVHN images.
The data is normalized and then 400 examples from the training data and
200 examples from the extra data are taken as test set. The rest of the
images is used for training. This should follow the practice from `Maxout
Networks <https://arxiv.org/abs/1302.4389>`_ as stated in `Densely
Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.
"""
from itertools import chain

import numpy
from lasagne.utils import floatX

from .svhn import FullSVHN


__all__ = ('Normalized', )


def sortout_indices(training, extra):
    """Sample images from the training and the extra data.

    This function will sample 400 images from the training data and 200
    images from the extra data for each class. After that the indices
    from the data sets are sorted and the length of the training set is
    added, as an offset, to the indices from the extra data.
    Those 6,000 images are used as test set, the rest is used for
    training.
    """

    def get_by_idx(labels, cls, cut, offset=0):
        """Sample a number of indices from a given class."""
        idx = numpy.argwhere(labels == cls).flatten()
        numpy.random.shuffle(idx)
        if offset:
            idx = idx + offset
        return list(idx[cut:]), list(idx[:cut])

    def sum_idx(iterable):
        """Merge and sort the indices."""
        return sorted(chain.from_iterable(iterable))

    tr_tr, tr_te = zip(*[get_by_idx(training, i, 400) for i in range(10)])
    ex_tr, ex_te = zip(*[get_by_idx(extra, i, 200, len(training))
                         for i in range(10)])
    return sum_idx(tr_tr) + sum_idx(ex_tr), sum_idx(tr_te) + sum_idx(ex_te)


class Normalized(FullSVHN):
    """The Street View House Numbers data set.

    Street View House Numbers data set consist of 32x32 color images
    depicting the numbers from 1 to 10. Each image shows a number in
    it's center and may also a distracting number at either sides.

    This class provides as normalized version of that data. The training
    set is sampled from the original training and extra examples.
    There are 400 examples from the training images and 200 examples from
    the extra images from each class. Those 6,000 images are used as test
    set, all the other images are used for training.

    This class provides a training set with 598,388 images, a test set with
    6,000 images and a validation set with 26,032 images.

    Parameters
    ----------
    root : string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite : boolean (``False``)
        If ``True`` any existing data will be overwritten.

    Note: The ``interval`` and ``testsplit`` parameters are removed due
    to the global contrast normalization and the sampling.

    For more information please see the `web site
    <http://ufldl.stanford.edu/housenumbers/>`_.
    """

    def __init__(self, testsplit=None, interval=None,
                 root='./_datasets', overwrite=False):
        if testsplit is not None:
            raise ValueError(
                'The test set is mandatory and can not be explicitly defined')
        if interval is not None:
            raise ValueError(
                'Normalization does not support setting an interval.')

        dct = self.load(root=root, overwrite=overwrite)
        idx_train, idx_test = sortout_indices(dct['original training labels'],
                                              dct['extra labels'])
        all_data = floatX(numpy.concatenate((dct['original training data'],
                                             dct['extra data'])))
        all_labels = numpy.concatenate((dct['original training labels'],
                                        dct['extra labels']))
        val_data = floatX(dct['validation data'])
        val_labels = dct['validation labels']
        del dct

        mean = numpy.mean(all_data, axis=(0, 2, 3), keepdims=True)
        std = numpy.std(all_data, axis=(0, 2, 3), keepdims=True)
        all_data = (all_data - mean) / std
        val_data = (val_data - mean) / std

        self.training = self.training_class(all_data[idx_train],
                                            all_labels[idx_train])
        self.test = self.test_class(all_data[idx_test], all_labels[idx_test])
        self.validation = self.test_class(val_data, val_labels)
