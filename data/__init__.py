"""This module provides some data set driven utilities.

The module provides some functions for creating/downloading and
benchmarking data sets. Those data sets usually consist of a training
and a validation set. The utilities here also provide a method to create
a test set by taking samples from the training set.

The classes (by default) download all necessary files into a given
directory, they also write caching files into there.
"""
from os import makedirs
from os.path import exists, join
from subprocess import run

import numpy
import h5py
from lasagne.utils import floatX
from theano import config


__all__ = ('change_interval', 'download', 'split', 'DataSet')


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


def change_interval(data, old=None, new=(0.0, 1.0)):
    """Move a matrix from one interval into another.

    Parameters
    ----------
    data : ``numpy.array``
       The data points.
    old : tuple (pair) of numbers or ``None`` (``None``)
        The old interval for the data. If it is ``None`` the minimum and
        maximum are taken from the input.
    new : tuple (pair) of numbers (``(0.0, 1.0)``)
        The new interval for the data.

    Returns
    -------
    ``numpy.array``
        The data points in the new interval.
    """
    n_min, n_max = new
    if old is None:
        o_min, o_max = None, None
    else:
        o_min, o_max = old
    if o_min is None:
        o_min = numpy.min(data)
    if o_max is None:
        o_max = numpy.max(data)

    if o_min >= o_max or n_min >= n_max:
        raise ValueError('The interval should be given as tuple of lower and'
                         ' upper bound.')

    return (data - o_min) / (o_max - o_min) * (n_max - n_min) + n_min


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
    if not (1 <= split < length):
        raise ValueError('The given split has an invalid value.')
    idx = numpy.random.permutation(length)
    idx_train = idx[:-split]
    idx_test = idx[-split:]
    return data[idx_train], labels[idx_train], data[idx_test], labels[idx_test]


def _load_dict_npz(path):
    """Load a dictionary of numpy arrays form a numpy (npz) file.

    Parameters
    ----------
    path : string
        The path of the file.

    Returns
    -------
    dictionary
        A dictionary of numpy arrays from the file.
    """
    return dict(numpy.load(path))


def _save_dict_npz(dct, path, compression=False):
    """Save a dictionary of numpy arrays into a (npz) file.

    Parameters
    ----------
    dct : dictionary
        The dictionary to save.
    path : string
        The path of the file.
    compression : boolean (``False``)
        If ``True`` the data will be saved in compressed form.
    """
    if compression:
        numpy.savez_compressed(path, **dct)
    else:
        numpy.savez(path, **dct)


def _h5_load(obj):
    """Load data from a :class:`h5py.Group` or a :class:`h5py.DataSet`."""
    if isinstance(obj, h5py.Group):
        return {k: _h5_load(v) for k, v in obj.items()}
    elif isinstance(obj, h5py.Dataset):
        return obj.value
    else:
        raise ValueError(f'Can not handle object "{obj}".')


SIZE_THRESH = h5py.h5z.SZIP_MAX_PIXELS_PER_BLOCK \
              / numpy.dtype(config.floatX).itemsize


def _h5_save(group, dct, **kwargs):
    """Save the data into a HDF5 structure."""
    for key, value in dct.items():
        if isinstance(value, dict):
            child = group.create_group(key)
            _h5_save(child, value, **kwargs)
        elif isinstance(value, numpy.ndarray) and value.ndim >= 1\
                and numpy.prod(value.shape) >= SIZE_THRESH:
            group.create_dataset(key, data=value, **kwargs)
        else:
            group.create_dataset(key, data=value)


def _load_dict_h5(path):
    """Load a dictionary of numpy arrays form a HDF5 (h5) file.

    Parameters
    ----------
    path : string
        The path of the file.

    Returns
    -------
    dictionary
        A dictionary of numpy arrays from the file.
    """
    with h5py.File(path) as fobj:
        return _h5_load(fobj)


def _save_dict_h5(dct, path, **kwargs):
    """Save a dictionary of numpy arrays in a HDF5 (h5) file.

    Parameters
    ----------
    dct : dictionary
        The dictionary to save.
    path : string
        The path of the file.
    kwargs : keyword arguments
        Keyword arguments that are passed down to all ``create_dataset``
        class made in the process. Some examples are ``chunks``,
        ``compression``, ``compression_opts``, ``shuffle`` and ``fletcher32``.
    """
    if not isinstance(dct, dict):
        raise ValueError(f'Parameter `dct` has the wrong type "{type(dct)}".')
    with h5py.File(path) as fobj:
        _h5_save(fobj, dct, **kwargs)


def _get_fileformat(path, fileformat, allowed=('npz', 'h5')):
    """Pass or guess the file format."""
    if not fileformat:
        fileformat = path.rsplit('.', 1)[1]
    if fileformat not in allowed:
        raise ValueError(f'Unsupported file format "{fileformat}".')
    return fileformat


def load_dict(path, fileformat=None):
    """Load a dictionary from a file.

    Parameters
    ----------
    path : string
        The path to the file.
    fileformat : ``'npz'``, ``'h5'``, or ``None`` (``None``)
        The file format to use. In case of ``None`` the format is guessed
        from the file name.

    Returns
    -------
    dictionary
        A dictionary with the data from the file.
    """
    fileformat = _get_fileformat(path, fileformat)
    if fileformat == 'h5':
        return _load_dict_h5(path)
    return _load_dict_npz(path)


def save_dict(dct, path, fileformat=None, **kwargs):
    """Save a dictionary into a file.

    The keyword arguments are optional and depend on the file format
    for numpy there is the ``compression`` parameter, which can be set to
    ``True`` in order to enable (zip) compression.
    For HDF5 there are following options:

    + ``compression``, which can be either ``'gzip'``, ``'lzf'`` or
      ``'szip'``.
    + ``compression_opts``, which is an integer from 0 to 9 and describes
      the compression rate
    + ``chunks``, can be set to ``True`` in order to enable auto-chunking
      or set to a chunk size.
    + ``shuffle``, can be set to ``True`` in order to rearrange the
      bytes in the chunk.
    + ``fletcher32``, can be set to ``True`` in order to add a checksum
      to each chunk.

    Parameters
    ----------
    dct : dictionary
        The dictionary with the data to save.
    path : string
        The file path.
    fileformat : ``'npz'``, ``'h5'``, or ``None`` (``None``)
        The file format to use. In case of ``None`` the format is guessed
        from the file name.
    kwargs : keyword arguments
        The format specific keyword arguments that get passed down.
    """
    fileformat = _get_fileformat(path, fileformat)
    if fileformat == 'h5':
        _save_dict_h5(dct, path, **kwargs)
    else:
        _save_dict_npz(dct, path, **kwargs)


class Data():
    """A set of labeled data.

    Parameters
    ----------
    data : ``numpy.array``
        The input data.
    labels : ``numpy.array`` or list
        The label information.
    """

    def __init__(self, data, labels):
        self._data = floatX(data)
        self._labels = numpy.array(labels, dtype=numpy.int32)

    @property
    def data(self):
        """The input data of the data set."""
        return self._data

    @property
    def labels(self):
        """The labels of the data set."""
        return self._labels

    @property
    def set(self):
        """The input data and labels of the data set."""
        return self.data, self.labels

    @property
    def state(self):
        """The current state of the data set as ``dict``."""
        return {'data': self.data, 'labels': self.labels}

    @classmethod
    def from_state(cls, state):
        """Create a new ``Data`` instance from a state."""
        return cls(**state)

    def _batch_iter_(self, batchsize, indices, truncate=False):
        """Iterate over mini-batches."""
        if not truncate and len(indices) % batchsize:
            raise ValueError('Length of the data set is not dividable by'
                             ' the batch size.')
        for idx in range(0, len(indices) - batchsize + 1, batchsize):
            batch = indices[idx:idx+batchsize]
            yield self.data[batch], self.labels[batch]


class TestData(Data):
    """A set of labeled test data.

    Parameters
    ----------
    data : ``numpy.array``
        The input data.
    labels : ``numpy.array`` or list
        The label information.
    """

    def iter(self, batchsize, indices=None):
        """Iterate over the data set in mini-batches.

        Parameters
        ----------
        batchsize : integer
            The number of data point per batch.
        indices : list or ``numpy.array`` of integers or ``None`` (``None``)
            Only use elements with these indices, if ``None`` all elements
            are used.

        Yields
        ------
        ``numpy.array``
            The data points of the current batch.
        ``numpy.array``
            The label information for the current batch.
        """
        if indices is None:
            indices = numpy.arange(len(self.data))
        yield from self._batch_iter_(batchsize, indices)


class TrainingData(Data):
    """A set of labeled training data.

    Parameters
    ----------
    data : ``numpy.array``
        The input data.
    labels : ``numpy.array`` or list
        The label information.
    random_state : tuple or ``None`` (``None``)
        The random state used for shuffling the data.
    """

    def __init__(self, data, labels, random_state=None):
        super(TrainingData, self).__init__(data, labels)
        self.random = numpy.random.RandomState()
        if random_state:
            self.random.set_state(random_state)

    def iter(self, batchsize, indices=None):
        """Iterate over mini-batches of the training data.

        Parameters
        ----------
        batchsize : integer
            The number of data point per batch.
        indices : list or ``numpy.array`` of integers or ``None`` (``None``)
            Only use elements with these indices, if ``None`` all elements
            are used.

        Yields
        ------
        ``numpy.array``
            The data points of the current batch.
        ``numpy.array``
            The label information for the current batch.
        """
        if indices is None:
            indices = numpy.arange(len(self.data))
        indices = self.random.permutation(indices)
        yield from self._batch_iter_(batchsize, indices, truncate=True)

    def endless_iter(self, batchsize, indices=None):
        """Iterate endlessly over the training data.

        Parameters
        ----------
        batchsize : integer
            The number of elements in a batch.
        indices : list or ``numpy.array`` of integers or ``None`` (``None``)
            Only use elements with these indices, if ``None`` all elements
            are used.

        Yields
        ------
        ``numpy.array``
            The data points of the current batch.
        ``numpy.array``
            The label information for the current batch.
        """
        if indices is None:
            indices = numpy.arange(len(self.data))
        todos = []
        while True:
            todos.extend(self.random.permutation(indices))
            length = (len(todos) // batchsize) * batchsize
            yield from self._batch_iter_(batchsize, todos[:length])
            todos = todos[length:]

    @property
    def state(self):
        return {'data': self.data, 'labels': self.labels,
                'random_state': self.random.get_state()}


class DataSet():
    """The basic super class for all data sets.

    A basic :class:`DataSet` has a training, an optional test and a
    validation set. The test set can be sampled from the training set or
    be set to the validation set.
    The class downloads and processes all required files autonomously
    and caches them in a given directory.

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
    """

    training_class = TrainingData
    test_class = TestData

    def __init__(self, testsplit=None, interval=(0, 1),
                 root='./_datasets', overwrite=False):
        dct = self.load(root=root, overwrite=overwrite)
        if interval is None:
            self.validation = self.test_class(dct['validation data'],
                                              dct['validation labels'])
        else:
            self.validation = self.test_class(
                change_interval(dct['validation data'], new=interval),
                dct['validation labels']
            )

        data, labels = dct['training data'], dct['training labels']
        if interval is not None:
            data = change_interval(data, new=interval)
        if not testsplit:
            self.training = self.training_class(data, labels)
            self.test = None
        elif testsplit == 'validation':
            self.training = self.training_class(data, labels)
            self.test = self.validation
        else:
            dtr, ltr, dte, lte = split(data, labels, testsplit)
            self.training = self.training_class(dtr, ltr)
            self.test = self.test_class(dte, lte)

    def load(self, root='./_datasets', overwrite=False):
        """Create and load the dataset (as ``dict``) from the files.

        This method loads a new data set from the files. This may include
        downloading necessary files. In some cases the data set is also
        loaded from a cached file.

        Parameters
        ----------
        root : string (``'./_datasets'``)
            Path to the root directory. This directory is used to save
            downloaded and caching files.
        overwrite : boolean (``False``)
            If ``True`` the cached files will be overwritten.

        Returns
        -------
        ``DataSet``
            The newly loaded data set.
        """
        makedirs(root, exist_ok=True)
        self.download(root=root)

        cache_file = getattr(self, 'cache_file', False)
        if cache_file:
            path = join(root, cache_file)
            if exists(path) and not overwrite:
                return load_dict(path)
            dct = self.extract(root=root)
            save_dict(dct, path)
            return dct
        else:
            return self.extract(root=root)

    @property
    def state(self):
        """The current state of the data set as a dictionary."""
        result = {
            'training': self.training.state,
            'validation': self.validation.state,
        }

        if self.test is self.validation:
            result['test'] = 'validation'
        elif self.test is not None:
            result['test'] = self.test.state

        return result

    def _save_state_npz(self, path, compression=False):
        """Save the current state into a numpy (.npz) file."""
        _save_dict_npz(self.state, path, compression=compression)

    def _save_state_h5(self, path, **kwargs):
        """Save the current state into a HDF5 (.h5) file."""
        state = self.state
        for dataset in state.values():
            if 'random_state' in dataset:
                rng = dataset['random_state']
                dataset['random_state'] = {f'{i}': s
                                           for i, s in enumerate(rng)}
        _save_dict_h5(state, path, **kwargs)

    def save_state(self, path, fileformat=None, **kwargs):
        """Save the current state into a file.

        The keyword arguments are optional and depend on the file format
        for numpy there is the ``compression`` parameter, which can be set to
        ``True`` in order to enable (zip) compression.
        For HDF5 there are following options:

        + ``compression``, which can be either ``'gzip'``, ``'lzf'`` or
          ``'szip'``.
        + ``compression_opts``, which is an integer from 0 to 9 and describes
          the compression rate
        + ``chunks``, can be set to ``True`` in order to enable auto-chunking
          or set to a chunk size.
        + ``shuffle``, can be set to ``True`` in order to rearrange the bytes
          in the chunk.
        + ``fletcher32``, can be set to ``True`` in order to add a checksum to
          each chunk.

        Parameters
        ----------
        path : string
             The file path.
        fileformat : ``'npz'``, ``'h5'`` or ``None`` (``None``)
            The file format to use. In case of ``None`` it will be guessed
            from the file extension.
        kwargs : keyword arguments
            The file format specific keyword arguments.
        """
        fileformat = _get_fileformat(path, fileformat)
        if fileformat == 'h5':
            self._save_state_h5(path, **kwargs)
        else:
            self._save_state_npz(path, **kwargs)

    @staticmethod
    def _read_state_npz(path):
        """Read a state from a numpy (.npz) file."""
        return {k: v.item() for k, v in _load_dict_npz(path).items()}

    @staticmethod
    def _read_state_h5(path):
        """Read a state from a HDF5 (.h5) file."""
        state = _load_dict_h5(path)
        for dataset in state.values():
            if 'random_state' in dataset:
                dct = dataset['random_state']
                dataset['random_state'] = tuple(dct[f'{i}']
                                                for i in range(len(dct)))
        return state

    @classmethod
    def from_state(cls, path_or_dict, fileformat=None):
        """Load a ``DataSet`` instance from a state.

        Parameters
        ----------
        path_or_dict : string or dictionary
            The state of the data set in form of a dictionary or a file path.
        fileformat : ``'npz'``, ``'h5'`` or ``None`` (``None``)
            The file format to use. In case of ``None`` it will be guessed
            from the file extension. This argument will be ignored in case
            ``path_or_dict`` is a dictionary.

        Returns
        -------
        ``DataSet``
            The newly loaded data set.
        """
        if isinstance(path_or_dict, str):
            fileformat = _get_fileformat(path_or_dict, fileformat)
            if fileformat == 'h5':
                dct = cls._read_state_h5(path_or_dict)
            else:
                dct = cls._read_state_npz(path_or_dict)
        elif isinstance(path_or_dict, dict):
            dct = path_or_dict
        else:
            raise ValueError('Can not handle the type of state:'
                             f' "{path_or_dict}".')

        result = cls.__new__(cls, {})
        result.training = cls.training_class.from_state(dct['training'])
        result.validation = cls.test_class.from_state(dct['validation'])
        if 'test' not in dct:
            result.test = None
        elif dct['test'] == 'validation':
            result.test = result.validation
        else:
            result.test = cls.test_class.from_state(dct['test'])
        return result

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

    def extract(self, root='./_datasets'):
        """Extract the data from the downloaded source file(s).

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
