Utility module for training lasagne models
==========================================

This repository contains multiple tools to train `Lasagne
<https://github.com/Lasagne/Lasagne>`_ models for image classification.
Included are some benchmark data sets, trainer classes, convenience functions
for loading and saving model parameters and plotting.


Installation
------------
This whole repository is designed to be installed via:

.. code-block:: bash

  git submodule add <repo>/imgcls_utils utils

After that the repository is used as python module.


Data Sets
---------
The ``data`` module contains several benchmark data sets for image
classification. Currently included are:

* `MNIST <http://yann.lecun.com/exdb/mnist/>`_
* `CIFAR-10 and CIFAR-100
  <https://www.cs.toronto.edu/~kriz/cifar.html>`_
* `The Street View House Numbers (SVHN)
  <http://ufldl.stanford.edu/housenumbers/>`_
* An augmented version of CIFAR-10 and CIFAR-100, where the training images
  where all zero-padded with 4 pixel at each side and the randomly cropped to
  the original size. Further are the training images horizontally flipped at
  random.
* A distractor data set constructed from MNIST. All the images show two numbers
  beside each other. If one of the numbers on the image is a number from 0-4
  the label will be ``1``, otherwise ``0``.

All data set classes take care of the download and pre-processing of the data
and cache the results in a given directory. The downloading is done via either
``wget`` or ``curl``.
By default all files, are stored in the directory ``./_datasets``. This can be
changed by passing a new path via the ``root`` parameter.

.. code-block:: python

    from data.mnist import MNIST

    dataset = MNIST(root='/tmp/data')


Training
--------
The ``training`` module contains two classes for training. One trains the
network iteration based, the other one in epochs.
Both trainer train a model for a set number of iterations or epochs with a given
learning rate.
They also feature methods to save the state of the trainer into a file, in order
to halt and continue the training.


Example Code
------------
.. code-block:: python

  from lasagne.layers import get_all_layers, get_all_params
  from lasagne.regularization import l2, regularize_layer_params
  from lasagne.updates import momentum

  from utils import loss_acc, save_model
  from utils.data.cifar_lee14 import CIFAR10
  from utils.training import IterationTrainer

  class Trainer(IterationTrainer):

    def __init__(self, *args, batchsize=128, **kwargs):
        super(Trainer, self).__init__(*args, batchsize=batchsize, **kwargs)
        self.dataset = CIFAR10()

        input_var = tensor.tensor4('inputs')
        target_var = tensor.ivector('targets')

        loss, _ = loss_acc(self.model, input_var, target_var,
                           deterministic=False)
        layers = get_all_layers(self.model)
        decay = regularize_layer_params(layers, l2) * 0.0001
        loss = loss + decay

        params = get_all_params(self.model, trainable=True)
        updates = momentum(loss, params, momentum=0.9,
                           learning_rate=self.learn_rate)

        self.set_training(input_var, target_var, loss, updates)


  model = ...
  trainer = Trainer(model)
  trainer.train([{'iterations': 32000, 'learning rate': 0.1},
                 {'iterations': 16000, 'learning rate': 0.01},
                 {'iterations': 16000, 'learning rate': 0.001}])

  _, acc = trainer.validate()
  save_model(model, 'model_with_{:0.2f}_acc.npz'.format(acc * 100))


Additional Data Sets
--------------------
If you want to add your own data set you have to implement a ``download``
method, that downloads all required files and a ``create`` method, that
loads the data from the files and returns a ``dict`` with the fields
``'training data'``,  ``'training labels'``, ``'validation data'`` and
``'validation labels'``.

.. code-block:: python

   from os.path import join
   import pickle

   from data import DataSet, download


   class MyDataSet(DataSet):

       @staticmethod
       def download(root='./_datasets', overwrite=False):
           download('www.example.com/mydataset.pkl', join(root, 'mydata.pkl'),
                   overwrite=overwrite)

       def create(self, root='./_datasets'):
           with open(join(root, 'mydata.pkl')) as fobj:
               return pickle.load(fobj)
