"""Plotting utilities.

This module contains several utilities for plotting using the
:mod:`pylab` and :mod:`matplotlib` packages. This includes
functions/iterators to create nested subplots and a normalizer
to create heatmaps.

The following code

>>> outer = hsplit(2, 1)
>>> next(outer)
>>> plot_img()
>>> inner = vsplit(1, 1,parent=next(outer))
>>> next(inner)
>>> plot_img()
>>> next(inner)
>>> plot_img()

will create a plot with the following layout

::
    +-------+
    | *   * |
    | *   * |
    +---+---+
    | * | * |
    +---+---+


This example will create a 10x10 plot with images

>>> for image, _ in square_griter(images[:100]):
>>>     plot_img(image)
"""
from collections import namedtuple
from itertools import product
from math import ceil

import numpy
from matplotlib import pylab
from matplotlib.colors import Normalize


# from http://matplotlib.org/users/colormapnorms.html
class MidpointNormalize(Normalize):
    """Normalize the negative and positive data differently.

    This class is a normalizer for plotting, usually passed as ``norm``
    parameter to the ``plot`` function. The code was taken from the
    `matplotlib's User’s Guide
    <http://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges>`_

    It is mostly used for heatmaps.
    """
    def __init__(self, vmin=None, vmax=None, midpoint=0.0, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return numpy.ma.masked_array(numpy.interp(value, x, y))


Parent = namedtuple('Parent', ['rows', 'columns', 'row_pos', 'col_pos',
                               'width', 'height', 'axis'])


def iter_grid(rows, columns, parent=None):
    """Iterate over (nested) subplots.

    The iterator will call ``pylab``'s subplot function and yield a
    object to create nested subplots. If such a object is passed as
    parent parameter, the iterator will create subplots in it's
    parent-plots.

    Parameters
    ----------
    rows : integer
        The number of rows in the grid.
    columns : integer
        The number of columns in the grid-
    parent : a :class:`Parent` instance or ``None`` (``None``)
        If not ``None``, then this is an object to generate
        nested subplots.

    Yields
    ------
    a :class:`Parent` instance
        An object containing information for generating subplot via this
        module's functions. It also has an attribute called ``axis``
        that contains the result of the subplot function.
    """
    if parent is None:
        shape = rows, columns
        width, height = 1, 1
        off_row, off_col = 0, 0
    else:
        shape = parent.rows * rows, parent.columns * columns
        width, height = parent.width, parent.height
        off_row, off_col = parent.row_pos * rows, parent.col_pos * columns

    for i_row, i_col in product(range(rows), range(columns)):
        row_pos = off_row + i_row * width
        col_pos = off_col + i_col * height
        axis = pylab.subplot2grid(shape, loc=(row_pos, col_pos))
        yield Parent(*shape, row_pos, col_pos, width, height, axis)


def square_griter(iterable, size=None, parent=None):
    """Try to make a quadratic grid for an iterator.

    Parameters
    ----------
    iterable : iterable
        An iterable object to build a grid for.
    size : tuple (pair) of integers or ``None`` (``None``)
        The size of the grid.
    parent : a :class:`Parent` instance or ``None`` (``None``)
        If not ``None``, then this is an object to generate
        nested subplots.

    Yields
    ------
    an object
        An object from the iterable.
    a :class:`Parent` instance
        An object containing information for generating subplot via this
        module's functions. It also has an attribute called ``axis``
        that contains the result of the subplot function.
    """
    if size is None:
        iterable = tuple(iterable)
        length = len(iterable)
        root = ceil(numpy.sqrt(length))
        grid = iter_grid(root, ceil(length / root), parent=parent)
    else:
        grid = iter_grid(*size, parent=parent)

    yield from zip(iterable, grid)


def product_griter(row_iter, col_iter, parent=None):
    """Iterate over subplots, span by two iterables.

    Parameters
    ----------
    row_iter : iterable
        An iterable yielding an object for every row.
    col_iter : iterable
        An iterable yielding an object for every column.
    parent : a :class:`Parent` instance or ``None`` (``None``)
        If not ``None``, then this is an object to generate
        nested subplots.

    Yields
    ------
    an object
        An object from the row-iterable.
    an object
        An object from the column-iterable.
    a :class:`Parent` instance
        An object containing information for generating subplot via this
        module's functions. It also has an attribute called ``axis``
        that contains the result of the subplot function.
    """
    rows = tuple(row_iter)
    columns = tuple(col_iter)
    griter = iter_grid(len(rows), len(columns), parent=parent)
    for grid, (row, colum) in zip(griter, product(row_iter, col_iter)):
        yield row, colum, grid


def vsplit(*splits, parent=None):
    """Vertically split a plot with different sizes.

    Parameters
    ----------
    splits : number of integers
        Each number describes the width of the column.
        The iterator ``vsplit(2, 1)`` will yield two subplots, where
        the first is left to the second one and is twice as big.
    parent : a :class:`Parent` instance or ``None`` (``None``)
        If not ``None``, then this is an object to generate
        nested subplots.

    Yields
    ------
    a :class:`Parent` instance
        An object containing information for generating subplot via this
        module's functions. It also has an attribute called ``axis``
        that contains the result of the subplot function.
    """
    columns = sum(splits)
    if parent is None:
        shape = 1, columns
        width, height = 1, 1
        off_row, off_col = 0, 0
    else:
        shape = parent.rows, parent.columns * columns
        width, height = parent.width, parent.height
        off_row, off_col = parent.row_pos, parent.col_pos * columns

    location = 0
    for split in (s * width for s in splits):
        col_pos = off_col + location
        axis = pylab.subplot2grid(shape, loc=(off_row, col_pos),
                                  colspan=split, rowspan=height)
        yield Parent(*shape, off_row, col_pos, split, height, axis)
        location += split


def hsplit(*splits, parent=None):
    """Horizontally split a plot with different sizes.

    Parameters
    ----------
    splits : number of integers
        Each number describes the height of the rows.
        The iterator ``hsplit(2, 1)`` will yield two subplots, where
        the first is above the second and twice as big.
    parent : a :class:`Parent` instance or ``None`` (``None``)
        If not ``None``, then this is an object to generate
        nested subplots.

    Yields
    ------
    a :class:`Parent` instance
        An object containing information for generating subplot via this
        module's functions. It also has an attribute called ``axis``
        that contains the result of the subplot function.
    """
    rows = sum(splits)
    if parent is None:
        shape = rows, 1
        width, height = 1, 1
        off_row, off_col = 0, 0
    else:
        shape = parent.rows * rows, parent.columns
        width, height = parent.width, parent.height
        off_row, off_col = parent.row_pos * rows, parent.col_pos

    location = 0
    for split in (s * height for s in splits):
        row_pos = off_row + location
        axis = pylab.subplot2grid(shape, loc=(row_pos, off_col),
                                  colspan=width, rowspan=split)
        yield Parent(*shape, row_pos, off_col, width, split, axis)
        location += split
