"""This is an utility module for training lasagne models.

This module contains multiple tools to train `Lasagne
<https://github.com/Lasagne/Lasagne>`_ models for image classification.
Included are some benchmark data sets, trainer classes, convenience
functions for loading and saving model parameters and plotting.

This file contains the convenience functions for the models.
"""
from functools import wraps

import numpy
from lasagne.layers import get_all_param_values, get_output,\
    set_all_param_values
from lasagne.objectives import categorical_crossentropy
from theano import config, function, tensor


__all__ = ('load_model', 'save_model', 'load_updates', 'save_updates',
           'loss_acc', 'test_model', 'mini_batch_func')


def load_model(model, path):
    """Load a the model parameters from a file and set them.

    Parameters
    ----------
    model : a :class:`Layer` instance
        The model with unset parameters.
    path : string
        The file with the model parameters.

    Returns
    -------
    a :class:`Layer` instance
        The given model with set parameters.
    """
    with numpy.load(path) as fobj:
        values = [fobj['arr_%d' % i] for i in range(len(fobj.files))]
        set_all_param_values(model, values)
    return model


def save_model(model, path):
    """Save the model parameters to a file.

    Parameters
    ----------
    model : a :class:`Layer` instance
        The model to save.
    path : string
        The destination file path.
    """
    numpy.savez(path, *get_all_param_values(model))


def load_updates(updates, path):
    """Load the updates from a file.

    Parameters
    ----------
    updates : dictionary of theano variables
        The dictionary containing the updates for the network parameters.
    path : string
        The file with the parameter values,

    Returns
    -------
    dictionary of theano variables
        The dictionary with the variable-values set from the file.
    """
    with numpy.load(path) as fobj:
        for var, i in zip(updates.keys(), range(len(fobj.files))):
            var.set_value(fobj['arr_%d' % i])


def save_updates(updates, path):
    """Save the updates to file.

    Parameters
    ----------
    updates : dictionary of theano variables
        The dictionary containing the updates for the network parameters.
    path : string
        The destination file path.
    """
    numpy.savez(path, *(p.get_value() for p in updates.keys()))


def loss_acc(model, input_var, target_var, deterministic=True):
    """Calculate the loss/error and accuracy of a model.

    Parameters
    ----------
    model : a :class:`Layer` instance
        The model to evaluate.
    input_var : theano symbolic variable
        A variable representing the network input.
    target_var : theano symbolic variable
        A variable representing the desired network
        output.
    deterministic : boolean (``True``)
        Use deterministic mode (for testing) or not (for training).

    Returns
    -------
    theano symbolic variable (scalar)
        The categorical cross-entropy.
    theano symbolic variable (scalar)
        The accuracy.
    """
    prediction = get_output(model, inputs=input_var,
                            deterministic=deterministic)
    loss = categorical_crossentropy(prediction, target_var)
    acc = tensor.eq(tensor.argmax(prediction, axis=1), target_var)
    return tensor.mean(loss), tensor.mean(acc, dtype=config.floatX)


def test_model(model, x_test, y_test, batchsize=500):
    """Test the given model on the test set.

    Parameters
    ----------
    model : a :class:`Layer` instance
        The model to evaluate.
    x_test : ``numpy.array``
        The data used for evaluation.
    y_test : ``numpy.array`` or list
        The label information for the data.
    batchsize : integer (``500``)
        The batch size to use.

    Returns
    -------
    float
        The loss (categorical cross-entropy).
    float
        The accuracy.
    """
    input_var = tensor.tensor4('inputs')
    target_var = tensor.ivector('targets')

    loss, acc = loss_acc(model, input_var, target_var)
    val_fn = function([input_var, target_var], [loss, acc])
    val_fn = mini_batch_func(val_fn, batchsize)

    err, acc = val_fn(x_test, y_test)
    return err, acc


def mini_batch_func(func, batchsize, shuffle=False, mean=True):
    """Apply a (theano) function batch-wise on a data set.

    Parameters
    ----------
    func : callable
        The function to execute on batches.
    batchsize : integer
        The size of each batch.
    shuffle : boolean (``False``)
        If ``True`` the data will be shuffled before execution.
    mean : boolean (``True``)
        If ``True`` the function will return the mean of all results,
        otherwise they will be concatenated.

    Returns
    -------
    callable
        a function that executes ``func`` batch-wise
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper for the train- or test-function.

        Parameters
        ----------
        args : list of ``numpy.array``
            A list of parameters. A batch from each array will be passed
            down to the wrapping function.
        kwargs : key-word arguments
            Key-word arguments that will be passed down to the wrapping
            function.
        """
        length = len(args[0])
        indices = numpy.arange(length)
        if shuffle:
            numpy.random.shuffle(indices)
        batches = [indices[i:i+batchsize] for i in range(0, length, batchsize)]
        lengths = [len(b) for b in batches]

        def post_process(result):
            """Concatenate the results and may calculate the mean."""
            result = numpy.array(result)
            if all(r.shape and r.shape[0] == l
                   for r, l in zip(result, lengths)):
                # function returned a scalar/vector per data-point
                result = numpy.concatenate(result)
                return numpy.mean(result, axis=0) if mean else result
            # function returned a scalar/vector per batch
            if not mean:
                return result
            result = [r * l for r, l in zip(result, lengths)]
            return sum(result) / sum(lengths)

        results = [func(*(a[batch] for a in args), **kwargs)
                   for batch in batches]
        if isinstance(results[0], list):
            return [post_process(x) for x in zip(*results)]
        return post_process(results)
    return wrapper
