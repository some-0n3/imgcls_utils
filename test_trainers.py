import os
import shutil
from tempfile import mkdtemp

import numpy
from lasagne.layers.conv import Conv2DLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers.helper import get_all_layers, get_all_params
from lasagne.layers.input import InputLayer
from lasagne.layers.noise import DropoutLayer
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers.pool import GlobalPoolLayer, Pool2DLayer
from lasagne.layers.special import NonlinearityLayer
from lasagne.nonlinearities import softmax
from lasagne.regularization import l2, regularize_layer_params
from lasagne.updates import momentum
from theano import tensor

from . import loss_acc
from .data.mnist import MNIST
from .training import EpochTrainer, IterationTrainer, Trainer


def example_network(dropout=True):
    model = InputLayer((None, 1, 28, 28))
    model = Pool2DLayer(model, 4, mode='average_inc_pad')

    def conv_layer(incoming, num_filters):
        tmp = Conv2DLayer(incoming, num_filters, 3, pad='valid')
        tmp = BatchNormLayer(tmp)
        if dropout:
            tmp = DropoutLayer(tmp, 0.3)
        return NonlinearityLayer(tmp)

    model = conv_layer(model, 64)
    model = conv_layer(model, 32)
    model = conv_layer(model, 16)

    model = GlobalPoolLayer(model)
    model = DenseLayer(model, 10, nonlinearity=softmax)
    return model


class TrainerMixin(Trainer):

    def __init__(self, *args, **kwargs):
        super(TrainerMixin, self).__init__(*args, **kwargs)
        input_var = tensor.tensor4('inputs')
        target_var = tensor.ivector('targets')

        loss, _ = loss_acc(self.model, input_var, target_var,
                           deterministic=False)
        layers = get_all_layers(self.model)
        decay = regularize_layer_params(layers, l2) * 0.0001
        loss = loss + decay

        params = get_all_params(self.model, trainable=True)
        updates = momentum(loss, params, momentum=0.9,
                           learning_rate=self.learning_rate)
        self.set_training(input_var, target_var, loss, updates)


def check_journals(baseline, journal):
    # TODO
    return True


class Base():

    def test_create(self):
        model = example_network()
        trainer = self.TrainerClass(model)
        assert trainer
        assert trainer.dataset is None

    def create(self, dataset=None, dropout=True):
        model = example_network(dropout=dropout)
        if dataset is None:
            dataset = MNIST()
        trainer = self.TrainerClass(model, dataset=dataset)
        return trainer

    def test_real_create(self):
        trainer = self.create()
        assert trainer
        assert trainer.dataset

    def test_simple_train(self):
        trainer = self.create()
        trainer.train(self.config)
        err, acc = trainer.validate()
        assert acc > 0.8
        assert trainer.journal

    def test_journal_file(self):
        trainer = self.create()
        tempdir = mkdtemp()
        prefix = os.path.join(tempdir, 'state')
        trainer.train(self.config)
        trainer.save_state(prefix)
        path = f'{prefix}_journal.pkl'
        copy = self.TrainerClass.load_journal(path)
        shutil.rmtree(tempdir)
        assert check_journals(trainer.journal, copy)

    def test_state_init(self):
        dataset = MNIST(testsplit=0.2)
        trainer = self.create(dataset=dataset)
        tempdir = mkdtemp()
        prefix = os.path.join(tempdir, 'state')

        trainer.save_state(prefix, resume=True)
        copy = self.TrainerClass.load_state(example_network(), prefix)
        shutil.rmtree(tempdir)

        trainer.train(self.config)
        copy.train(self.config)

        err1, acc1 = trainer.validate()
        err2, acc2 = copy.validate()
        assert numpy.allclose(err1, err2)
        assert numpy.allclose(acc1, acc2)
        assert err1 == err2
        assert acc1 == acc2

        assert check_journals(trainer.journal, copy.journal)

    def test_state(self):
        dataset = MNIST(testsplit=0.2)
        trainer = self.create(dataset=dataset)
        tempdir = mkdtemp()
        prefix = os.path.join(tempdir, 'state')

        trainer.train(self.config[0:1])

        trainer.save_state(prefix, resume=True)
        copy = self.TrainerClass.load_state(example_network(), prefix)
        shutil.rmtree(tempdir)

        trainer.train(self.config)
        copy.train(self.config)

        err1, acc1 = trainer.validate()
        err2, acc2 = copy.validate()
        assert numpy.allclose(err1, err2)
        assert numpy.allclose(acc1, acc2)
        assert err1 == err2
        assert acc1 == acc2

        assert check_journals(trainer.journal, copy.journal)

    def test_state_other(self):
        dataset = MNIST(testsplit=0.2)
        trainer = self.create(dataset=dataset)
        tempdir = mkdtemp()
        prefix = os.path.join(tempdir, 'state')

        trainer.train(self.config[0:1])

        trainer.save_state(prefix, resume=True)
        copy = self.TrainerClass.load_state(example_network(), prefix)

        trainer.save_state(os.path.join(tempdir, 'other'), resume=True)
        copy.save_state(os.path.join(tempdir, 'copy'), resume=True)

        shutil.rmtree(tempdir)

        trainer.train(self.config)
        copy.train(self.config)

        err1, acc1 = trainer.validate()
        err2, acc2 = copy.validate()
        assert numpy.allclose(err1, err2)
        assert numpy.allclose(acc1, acc2)
        assert err1 == err2
        assert acc1 == acc2

        assert check_journals(trainer.journal, copy.journal)

    def test_state_det(self):
        dataset = MNIST(testsplit=0.2)
        trainer = self.create(dataset=dataset, dropout=False)
        tempdir = mkdtemp()
        prefix = os.path.join(tempdir, 'state')
        trainer.save_state(prefix, resume=True)
        copy = self.TrainerClass.load_state(example_network(dropout=False),
                                            prefix)
        shutil.rmtree(tempdir)

        trainer.train(self.config)
        copy.train(self.config)

        err1, acc1 = trainer.validate()
        err2, acc2 = copy.validate()
        assert numpy.allclose(err1, err2)
        assert numpy.allclose(acc1, acc2)
        assert err1 == err2
        assert acc1 == acc2

        assert check_journals(trainer.journal, copy.journal)


class TestEpoch(Base):

    class TrainerClass(TrainerMixin, EpochTrainer):
        pass

    config = [{'learning rate': 0.1, 'epochs': 2},
              {'learning rate': 0.01, 'epochs': 1}]


class TestIteration(Base):

    class TrainerClass(TrainerMixin, IterationTrainer):
        pass

    config = [{'learning rate': 0.1, 'iterations': 500},
              {'learning rate': 0.01, 'iterations': 250}]
