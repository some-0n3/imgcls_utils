"""This module implements the training for lasagne models.

The classes are designed to train a lasagne model on a fixed data set
with different learning rates. The training schedule can be epoch or
iteration based.
The classes print the training and test error to the standard output
every epoch or in a given interval of iterations.

Please use the :class:`EpochTrainer` and :class:`IterationTrainer`
for training, the class :class:`Trainer` does not implements a
training mechanisms.

Example
-------
>>> input_var = ...
>>> target_var = ...
>>> model = ...
>>>
>>> trainer = EpochTrainer(model)
>>> trainer.dataset = MNIST(interval=(0, 1))
>>>
>>> prediction = get_output(model, inputs=input_var, deterministic=False)
>>> loss = categorical_crossentropy(prediction, target_var)
>>> params = get_all_params(model, trainable=True)
>>> updates = momentum(loss, params, momentum=0.9,
                       learning_rate=trainer.learning_rate)
>>> trainer.set_training(input_var, target_var, loss, updates)
>>>
>>> trainer.train_epochs(50, 0.001)
>>> trainer.train_epochs(20, 0.0001)
>>> save_model(trainer.model, 'mnist_model.npz')
"""
import os
import pickle
import time
from collections import OrderedDict
from warnings import warn

import numpy
from lasagne.layers.helper import get_all_layers
from lasagne.utils import floatX
from theano import function, shared, tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from . import load_model, load_updates, loss_acc, mini_batch_func, save_model,\
    save_updates
from .data import DataSet


__all__ = ('EpochTrainer', 'IterationTrainer')


def get_random_streams(model):
    """Return a list with all ``RandomStreams`` in the model."""
    return [l._srng for l in get_all_layers(model) if hasattr(l, '_srng')]


def save_random_streams(streams, path):
    """Save a list of random streams to a file."""
    dct = {}
    for i, stream in enumerate(streams):
        stream.seed()
        dct[f'seed_{i}'] = stream.default_instance_seed
    numpy.savez(path, **dct)


def load_random_streams(model, path):
    """Load the random streams from a file into a model."""
    layers = [l for l in get_all_layers(model) if hasattr(l, '_srng')]
    with numpy.load(path) as fobj:
        for i, layer in enumerate(layers):
            layer._srng = RandomStreams(fobj[f'seed_{i}'].item())


class Trainer():
    """The default super class for all trainers.

    This class implements basic mechanisms for the training, but not the
    training itself.

    Parameters
    ----------
    model : a :class:`Layer` instance
        The model to train.
    dataset : a :class:`Dataset` instance or ``None`` (``None``)
        The data set used for training and testing.
    batchsize : integer (``256``)
        The batch size used for training.
    val_batchsize : integer (``500``)
        The batch size used for testing and validation.
    """

    dataset_cls = DataSet

    def __init__(self, model, dataset=None, batchsize=256, val_batchsize=500):
        self.model = model
        self.dataset = dataset
        self.batchsize = batchsize
        self.val_bsize = val_batchsize
        self._learn_rate = shared(floatX(0.0))
        self.journal = []
        self.updates = None
        self._train = None
        self._value_names = []

        # compile validation function
        input_var = tensor.tensor4('inputs')
        target_var = tensor.ivector('targets')
        loss, acc = loss_acc(model, input_var, target_var)
        self._validate = function([input_var, target_var], [loss, acc])

    @property
    def learning_rate(self):
        """The learning rate used for training (shared variable)."""
        return self._learn_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        """Set the learning rate to the given value.

        Parameters
        ----------
        learning_rate : float
            The value for the learning rate.
        """
        # TODO : support other types (e.g. shared variable)
        self._learn_rate.set_value(floatX(learning_rate))

    def set_training(self, input_var, target_var, loss, updates, values=None):
        """Set the updates and compile the training function.

        Parameters
        ----------
        input_var : theano symbolic variable
            A variable representing the network input.
        target_var : theano symbolic variable
            A variable representing the desired network
            output.
        loss : scalar
            The theano expression for the loss.
        updates : OrderedDict
            The dictionary, mapping all of the network parameters
            to their update expressions.
        values : dictionary or ``None`` (``None``)
            A dictionary with additional values to be logged for the
            training. The dictionary should map names for the logging
            to their respective theano variables.
        """
        self.updates = updates
        logging = OrderedDict([('train_err', loss), ])
        if values is not None:
            logging.update(values)
        self._value_names = list(logging.keys())
        self._train = function([input_var, target_var], list(logging.values()),
                               updates=updates)

    def _batch_valid_(self, data, labels):
        # TODO : replace with `DataSet.iter`
        val_fn = mini_batch_func(self._validate, self.val_bsize)
        err, acc = val_fn(data, labels)
        return err, acc

    def test(self):
        """Return the loss and accuracy over the test set."""
        if self.dataset.test:
            return self._batch_valid_(*self.dataset.test.set)
        return floatX('nan'), floatX('nan')

    def validate(self):
        """Return the loss and accuracy over the validation set."""
        return self._batch_valid_(*self.dataset.validation.set)

    def save_state(self, prefix, resume=False):
        """Save the state of a trainer into 2 or 5 files.

        The trainer saves the parameter for the current model, the
        updates from the optimizer, state information about the
        random number generation, the journal and the dataset into 5
        separate files. All files will have the given prefix and
        different endings.

        Parameters
        ----------
        prefix : string
            The file name prefix for the files.
        resume : boolean (``False``)
            If ``True`` also save data needed for resuming the training.
            Otherwise only the model parameters and the journal is saved.
        """
        save_model(self.model, f'{prefix}.npz')
        with open(f'{prefix}_journal.pkl', 'wb') as fobj:
            pickle.dump(self.journal, fobj)
        if not resume:
            return
        save_updates(self.updates, f'{prefix}_updates.npz')
        if self.dataset:
            self.dataset.save_state(f'{prefix}_data.npz')
        streams = get_random_streams(self.model)
        if streams:
            save_random_streams(streams, f'{prefix}_rng_data.npz')

    @classmethod
    def load_state(cls, model, prefix, **kwargs):
        """Load the state of the trainer from files.

        This method will create a new trainer instance from the files
        with the given prefix. This method is to load a trainer that was
        saved via `save_state`.

        Parameters
        ----------
        model: a :class:`Layer` instance
            The (uninitialized) model.
        prefix: string
            The Prefix for the files.
        **kwargs : keyword arguments
            Key-word arguments that will be passed down to the
            constructor.

        Returns
        -------
        a :class:`Trainer` instance
            The new trainer with all the parameters loaded from the
            files.
        """
        # model parameters
        model = load_model(model, f'{prefix}.npz')

        # random stream data
        filename = f'{prefix}_rng_data.npz'
        if os.path.isfile(filename):
            load_random_streams(model, filename)

        trainer = cls(model, **kwargs)

        # updates
        filename = f'{prefix}_updates.npz'
        if os.path.isfile(filename):
            if trainer.updates is None:
                warn('Could not load update parameters.')
            else:
                load_updates(trainer.updates, filename)

        # journal
        with open(f'{prefix}_journal.pkl', 'rb') as fobj:
            trainer.journal = pickle.load(fobj)

        # dataset
        filename = f'{prefix}_data.npz'
        if os.path.isfile(filename):
            trainer.dataset = cls.dataset_cls.from_state(filename)

        return trainer

    def train(self, *args, **kwargs):
        """Train the network according to a given schedule."""
        raise NotImplementedError('')

    @staticmethod
    def _log_error_(values, valid_err, valid_acc):
        """Print the training, val. loss and val. acc."""
        print(f'    Training Loss:          {values[0].mean():>10.6f}')
        print(f'    Validation Loss:        {valid_err:>10.6f}')
        print(f'    Validation Accuracy: {valid_acc:>10.2%}')

    @staticmethod
    def load_journal(path):
        """Load a journal from a given file path.

        Parameters
        ----------
        path : string
            path to the journal file.

        Returns
        -------
        dictionary
            A dictionary containing the following training information:

            ``'train_err'`` the training error (loss) for each iteration.

            ``'valid_err'`` the error (cross-entropy) on the test set.

            ``'valid_acc'`` the accuracy on the test set.

            ``'valid_itr'`` the number of iteration at any point in
                ``'valid_err'`` or ``'valid_acc'``.

            Additionally to that, there is also a ``numpy.array`` for
            every entry in the ``values`` parameter that was passed to
            the ``Trainer.set_training`` method. The array contains the
            corresponding values to the logged variables for every
            iteration.
        """
        raise NotImplementedError('')


class EpochTrainer(Trainer):
    """Implements a epoch based training scheduling.

    Parameters
    ----------
    model : a :class:`Layer` instance
        The model to train.
    dataset : a :class:`Dataset` instance or ``None`` (``None``)
        The data set used for training and testing.
    batchsize : integer (``256``)
        The batch size used for training.
    val_batchsize : integer (``500``)
        The batch size used for testing and validation.
    """

    def __init__(self, *args, **kwargs):
        super(EpochTrainer, self).__init__(*args, **kwargs)
        self._total_epochs = 0

    @classmethod
    def load_state(cls, model, prefix, **kwargs):
        trainer = super(EpochTrainer, cls).load_state(model, prefix, **kwargs)
        trainer._total_epochs = len(trainer.journal)
        return trainer

    def _train_epochs_(self, epochs, learning_rate):
        """Train the network for a given amount of epochs."""
        self.learning_rate = learning_rate

        def train_fn():
            iterator = self.dataset.training.iter(self.batchsize)
            results = [self._train(d, l) for d, l in iterator]
            return [numpy.array(r) for r in zip(*results)]

        print(f'Training {epochs:>3} epochs '
              f'with a learning rate of {learning_rate:.6f}.')

        done_epochs = len(self.journal) + 1
        for current in range(done_epochs, done_epochs + epochs):
            entry = {'learning rate': learning_rate,
                     'batchsize': self.batchsize, 'epoch': current}
            start_time = time.time()

            values = train_fn()
            entry.update(zip(self._value_names, values))
            valid_err, valid_acc = self.test()
            entry['valid_err'] = valid_err
            entry['valid_acc'] = valid_acc

            duration = time.time() - start_time
            self.journal.append(entry)
            print(f'Epoch {current:>3}/{self._total_epochs:>3}'
                  f' took {duration:.3f}s')
            self._log_error_(values, valid_err, valid_acc)

    def train_epochs(self, epochs, learning_rate, **kwargs):
        """Train the network for a given number of epochs.

        Parameters
        ----------
        epochs : integer
            The number of epochs to train.
        learning_rate : float
            The learning rate to use.
        """
        self._total_epochs += epochs
        start_time = time.time()
        self._train_epochs_(epochs, learning_rate, **kwargs)
        duration = time.time() - start_time
        print(f'Training done. Time:{duration:>15.3f}s \t({epochs} epochs,'
              f' ~{duration / epochs:.3f}s/epoch).')

    def train(self, config, **kwargs):
        """Train the network according to a given schedule.

        Parameters
        ----------
        config : list of dictionaries
            A learning schedule for training. It is represented as a
            list of dictionaries, each containing a ``'learning rate'``
            and a ``'epochs'`` field that state how many epochs are to
            be trained with that learning rate.
        """
        print('Start training...')
        num_epochs = sum(e['epochs'] for e in config)
        self._total_epochs += num_epochs
        start_time = time.time()
        for entry in config:
            self._train_epochs_(entry['epochs'], entry['learning rate'],
                                **kwargs)
        duration = time.time() - start_time
        print(f'Training done. Time: {duration:>15.3f}s \t({num_epochs}'
              f' epochs, ~{duration / num_epochs:.3f}s/epoch).')

    @staticmethod
    def load_journal(path):
        """Load a journal from a given file path.

        Parameters
        ----------
        path : string
            path to the journal file.

        Returns
        -------
        dictionary
            A dictionary containing the following training information:

            ``'train_err'`` the training error (loss) for each iteration.

            ``'valid_err'`` the error (cross-entropy) on the test set.

            ``'valid_acc'`` the accuracy on the test set.

            ``'valid_itr'`` the number of iteration at any point in
                ``'valid_err'`` or ``'valid_acc'``.

            ``'train_err_epoch'`` the training error by epoch
            (as 2d-array).

            Additionally to that, there are also two ``numpy.array``
            for every entry in the ``values`` parameter that was passed
            to the ``Trainer.set_training`` method. One is named after
            the entry in the parameter and contains the corresponding
            values to the theano variable for every iteration. The other
            has the additional postfix ``'_epoch'`` and contains the
            epoch wise entries just like ``'train_err_epoch'``.
        """
        with open(path, 'rb') as fobj:
            journal = pickle.load(fobj)
            assert all(a == b for a, b in zip((j['epoch'] for j in journal),
                                              range(1, 1 + len(journal))))

            train_err = numpy.array([j['train_err'] for j in journal])

            epochs, step = train_err.shape
            learningrates = [[j['learning rate'], ] * step for j in journal]
            batchsizes = [[j['batchsize'], ] * step for j in journal]
            result = {
                'train_err_epoch': train_err,
                'learning_rates_epoch': numpy.array(learningrates),
                'batchsizes_epoch': numpy.array(batchsizes),
                'train_err': numpy.hstack(train_err),
                'train_itr': numpy.arange(1, epochs * step + 1),
                'valid_acc': numpy.array([j['valid_acc'] for j in journal]),
                'valid_err': numpy.array([j['valid_err'] for j in journal]),
                'valid_itr': numpy.arange(1, epochs + 1) * step,
                'learning_rates': numpy.hstack(learningrates),
                'batchsizes': numpy.hstack(batchsizes),
            }

            keys = set()
            keys.update(*[j.keys() for j in journal])
            keys -= {'train_err', 'epoch', 'learning rate', 'batchsize',
                     'valid_acc', 'valid_err'}
            for key in keys:
                result[key + '_epoch'] = numpy.array([j[key] for j in journal])
                result[key] = numpy.hstack([j[key] for j in journal])

            return result


class IterationTrainer(Trainer):
    """Implements a iteration based training scheduling.

    Parameters
    ----------
    model : a :class:`Layer` instance
        The model to train.
    dataset : a :class:`Dataset` instance or ``None`` (``None``)
        The data set used for training and testing.
    batchsize : integer (``256``)
        The batch size used for training.
    val_batchsize : integer (``500``)
        The batch size used for testing and validation.
    """

    def __init__(self, *args, **kwargs):
        super(IterationTrainer, self).__init__(*args, **kwargs)
        self._total_iters = 0

    @classmethod
    def load_state(cls, model, prefix, **kwargs):
        trainer = super(IterationTrainer, cls).load_state(model, prefix,
                                                          **kwargs)
        trainer._total_iters = len(trainer.journal)
        return trainer

    def _train_iters_(self, iterations, learning_rate, tick=200):
        """Train the model for a given amount of iterations."""
        self.learning_rate = learning_rate
        iterator = self.dataset.training.endless_iter(self.batchsize)

        print(f'Training {iterations:>3} iterations'
              f' with a learning rate of {learning_rate:.6f}.')
        tmpl = 'Iteration {:>7}/{:>7} @ ~{:.3f} iterations/s'

        done_iters = len(self.journal) + 1
        start_time = time.time()
        for current in range(done_iters, done_iters + iterations):
            entry = {'learning rate': learning_rate,
                     'batchsize': self.batchsize,
                     'iteration': current}
            self.journal.append(entry)
            data, labels = next(iterator)
            values = self._train(data, labels)
            entry.update(zip(self._value_names, values))

            if current % tick == 0:
                duration = time.time() - start_time
                valid_err, valid_acc = self.test()
                entry['valid_err'] = valid_err
                entry['valid_acc'] = valid_acc
                print(tmpl.format(current, self._total_iters, tick / duration))
                self._log_error_(values, valid_err, valid_acc)
                start_time = time.time()

    def train_iters(self, iterations, learning_rate, **kwargs):
        """Train the model for a given number of iterations.

        Parameters
        ----------
        iterations : integer
            The number of iterations to train.
        learning_rate : float
            The learning rate to use for training.
        tick : integer (``200``)
            The network will be evaluated on the test set every
            ``tick`` iterations.
        """
        self._total_iters += iterations
        start_time = time.time()
        self._train_iters_(iterations, learning_rate, **kwargs)
        duration = time.time() - start_time
        print(f'Training done. Time:{duration:>15.3f}s \t({iterations} iters.,'
              f' ~{iterations / duration:.3f} iter/s).')

    def train(self, config, **kwargs):
        """Train the network according to a given schedule.

        Parameters
        ----------
        config : list of dictionaries
            A learning schedule for training. It is represented as a
            list of dictionaries, each containing a ``'learning rate'``
            and a ``'iterations'`` field that state how many iterations
            are to be trained with that learning rate.
        tick : integer (``200``)
            The network will be evaluated on the test set every
            ``tick`` iterations.
        """
        print("Start training...")
        iterations = sum(e['iterations'] for e in config)
        self._total_iters += iterations
        start_time = time.time()
        for entry in config:
            self._train_iters_(entry['iterations'], entry['learning rate'],
                               **kwargs)
        delta = time.time() - start_time
        print(f'Training done. Time:{delta:>15.3f}s \t({iterations} iters,'
              f' ~{iterations / delta:.3f} iter/s).')

    @staticmethod
    def load_journal(path):
        with open(path, 'rb') as fobj:
            journal = pickle.load(fobj)
            assert all(a == b for a, b in zip(
                (j['iteration'] for j in journal),
                range(1, 1 + len(journal))
            ))
            keys = set()
            keys.update(*[j.keys() for j in journal])
            keys -= {'valid_err', 'valid_acc'}

            result = {k: numpy.array([j[k] for j in journal]) for k in keys}
            result['train_itr'] = numpy.arange(1, len(journal) + 1)
            result['valid_acc'] = numpy.array([j['valid_acc'] for j in journal
                                               if 'valid_acc' in j])
            result['valid_err'] = numpy.array([j['valid_err'] for j in journal
                                               if 'valid_err' in j])
            result['valid_itr'] = numpy.array(
                [i for i, j in enumerate(journal, 1) if 'valid_acc' in j])
            return result
