"""Visualize a neural network as a (dot) graph.

This module provides functions and classes to generate dot graphs for a
given lasagne model or a list of layers. It provides e.g. the ability to
"draw" a model "to_notebook", pretty much exactly like `nolearn
<https://github.com/dnouri/nolearn>`_ via:

>>> from utils.visualize import draw_to_notebook, nolearn
>>> model = ...
>>> draw_to_notebook(model, nolearn)

The module allows more customization for the drawing of notebooks.
The functions ``draw_to_file`` and ``draw_to_notebook`` take the keyword
argument ``node_creator``
The module provides the following creation functions:

    ``nolearn`` draws as nolearn's implementation would and takes the
        same arguments.

    ``verbose_create`` creates a graph with a lot of information.

    ``format_create`` allows type specific node creation by using string
        formatting. The color of the node and a format string that is
        used to create the label are passed on as two dictionaries.

    ``default_create`` is the default creator and executes
        ``format_create`` with either a short format map or
        a more verbose one

The module also provides ways to create and pass type specific color maps
for the layers (with static colors). The function ``colors_from_cmap``
creates such a dictionary for the colors from a (matplotlib) colormap.

The ``ParamFormatter`` that is used by default also provides some
useful format specifiers for a :class:`Layer`'s attributes.
"""
from string import Formatter

from lasagne.layers import Layer, conv, get_all_layers, pool, recurrent
from lasagne.layers.conv import Conv1DLayer, Conv2DLayer, Conv3DLayer,\
    Deconv2DLayer, DilatedConv2DLayer, TransposedConv2DLayer,\
    TransposedConv3DLayer
from lasagne.layers.dense import DenseLayer, NINLayer
from lasagne.layers.embedding import EmbeddingLayer
from lasagne.layers.input import InputLayer
from lasagne.layers.local import LocallyConnected2DLayer
from lasagne.layers.merge import ConcatLayer, ElemwiseMergeLayer,\
    ElemwiseSumLayer
from lasagne.layers.noise import DropoutLayer, GaussianNoiseLayer
from lasagne.layers.normalization import BatchNormLayer,\
    LocalResponseNormalization2DLayer
from lasagne.layers.pool import FeaturePoolLayer, FeatureWTALayer,\
    GlobalPoolLayer, MaxPool1DLayer, MaxPool2DLayer, MaxPool3DLayer,\
    Pool1DLayer, Pool2DLayer, Pool3DLayer, SpatialPyramidPoolingLayer,\
    Upscale1DLayer, Upscale2DLayer, Upscale3DLayer
from lasagne.layers.recurrent import CustomRecurrentLayer, GRULayer, Gate,\
    LSTMLayer, RecurrentLayer
from lasagne.layers.shape import DimshuffleLayer, FlattenLayer, PadLayer,\
    ReshapeLayer, SliceLayer
from lasagne.layers.special import BiasLayer, ExpressionLayer, InverseLayer,\
    NonlinearityLayer, ParametricRectifierLayer, RandomizedRectifierLayer,\
    ScaleLayer, TPSTransformerLayer, TransformerLayer
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex
from pydotplus import Dot, Edge, Node


__all__ = ('draw_to_file', 'draw_to_notebook', 'pydot_graph', 'nolearn',
           'SHORT', 'VERBOSE', 'default_create', 'format_create',
           'ParamFormatter', 'verbose_create', 'DEFAULT_MAP',
           'colors_from_cmap', 'dot_escape')


# draw like nolearn does it

NOLEARN_COLORS = ('#4A88B3', '#98C1DE', '#6CA2C8', '#3173A2', '#17649B',
                  '#FFBB60', '#FFDAA9', '#FFC981', '#FCAC41', '#F29416',
                  '#C54AAA', '#E698D4', '#D56CBE', '#B72F99', '#B0108D',
                  '#75DF54', '#B3F1A0', '#91E875', '#5DD637', '#3FCD12')


def _nolearn_color(layer):
    """Return a color for the given layer, like nolearn would."""
    cls_name = type(layer).__name__
    hashed = hash(cls_name) % 5
    if cls_name in conv.__all__:
        return NOLEARN_COLORS[:5][hashed]
    elif cls_name in pool.__all__:
        return NOLEARN_COLORS[5:10][hashed]
    elif cls_name in recurrent.__all__:
        return NOLEARN_COLORS[10:15][hashed]
    return NOLEARN_COLORS[15:20][hashed]


def nolearn(layer, output_shape=True, verbose=False, **kwargs):
    """Create a :class:`Node` for a given layer, like nolearn would.

    Parameters
    ----------
    layer : a class:`Layer` instance
        The layer for which a node shall be created.
    output_shape : boolean (``True``)
        If ``True`` the output shape of the layer will be displayed.
    verbose : boolean (''False`)
        If ``True`` layer attributes like filter shape, stride, etc.
        will be displayed.
    kwargs : keyword arguments
        Those will be passed down to :class:`Node`.
    """
    label = type(layer).__name__
    color = _nolearn_color(layer)
    if verbose:
        for attr in ['num_filters', 'num_units', 'ds', 'filter_shape',
                     'stride', 'strides', 'p']:
            if hasattr(layer, attr):
                label += f'\n{attr}: {getattr(layer, attr)}'
        if hasattr(layer, 'nonlinearity'):
            try:
                nonlinearity = layer.nonlinearity.__name__
            except AttributeError:
                nonlinearity = layer.nonlinearity.__class__.__name__
            label += f'\nnonlinearity: {nonlinearity}'

    if output_shape:
        label += f'\nOutput shape: {layer.output_shape}'

    return Node(repr(layer), label=label, shape='record',
                fillcolor=color, style='filled', **kwargs)


# get colors from a colormap

def _types_from_lasange():
    """Retrieve a list of all layer types from lasagne."""
    from lasagne.layers import input, base, dense, noise, local, shape, \
        merge, normalization, special
    modules = (input, base, conv, dense, recurrent, pool, shape, merge,
               normalization, noise, local, special)
    types = []
    for mod in modules:
        for name in mod.__all__:
            obj = getattr(mod, name)
            if obj in types:
                continue
            if isinstance(obj, type) and issubclass(obj, Layer):
                types.append(obj)
    return types


def colors_from_cmap(types=None, color_map='terrain'):
    """Create a color dict from a color map.

    Parameters
    ----------
    layers : list of :class:`Layer` instances or ``None`` (``None``)
        The color dict will be created for this list of layers, if
        ``None`` a list of all layers is retrieved from lasagne.
    color_map : string or colormap (``'terrain'``)
        The colormap to use.
    """
    types = types or _types_from_lasange()
    cmap = get_cmap(color_map, 2 + len(types) * 1.1)
    return {t:  rgb2hex(cmap(i)[:3]) for i, t in enumerate(types)}


DEFAULT_MAP = colors_from_cmap()


def dot_escape(obj):
    """Create a string a escape all illegal characters."""
    def replace_all(string, old):
        result = string
        for char in old:
            result = result.replace(char, '\\' + char)
        return result
    return replace_all(str(obj), '<>(){}-[]')


def verbose_create(layer, color_map=DEFAULT_MAP,
                   blacklist=('input_layer', 'input_layers'), **kwargs):
    """Create a node for the layer with a lot of information.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The layer.
    color_map ; dictionary
        A dictionary that maps all layer types to a color value.
    blacklist : sequence of strings
        A list of attribute names that are not included.
    kwargs : keyword arguments
        Those will be passed down to :class:`Node`.
    """
    label = type(layer).__name__
    color = color_map[type(layer)]
    variables = vars(layer)
    label += '\n' + '\n'.join((f'{n} : {dot_escape(variables[n])}'
                               for n in sorted(variables)
                               if n not in blacklist))
    return Node(repr(layer), label=label, shape='record',
                fillcolor=color, style='filled', **kwargs)


# create nodes via a class specific string format system

EMPTY = {
    # input
    InputLayer: '',
    # dense
    DenseLayer: '',
    NINLayer: '',
    # convolution
    Conv1DLayer: '',
    Conv2DLayer: '',
    Conv3DLayer: '',
    TransposedConv2DLayer: '',
    TransposedConv3DLayer: '',
    Deconv2DLayer: '',
    DilatedConv2DLayer: '',
    # local
    LocallyConnected2DLayer: '',
    # pooling
    Pool1DLayer: '',
    Pool2DLayer: '',
    Pool3DLayer: '',
    MaxPool1DLayer: '',
    MaxPool2DLayer: '',
    MaxPool3DLayer: '',
    Upscale1DLayer: '',
    Upscale2DLayer: '',
    Upscale3DLayer: '',
    GlobalPoolLayer: '',
    FeaturePoolLayer: '',
    FeatureWTALayer: '',
    SpatialPyramidPoolingLayer: '',
    # recurrent
    CustomRecurrentLayer: '',
    RecurrentLayer: '',
    LSTMLayer: '',
    GRULayer: '',
    Gate: '',
    # noise
    DropoutLayer: '',
    GaussianNoiseLayer: '',
    # shape
    ReshapeLayer: '',
    FlattenLayer: '',
    DimshuffleLayer: '',
    PadLayer: '',
    SliceLayer: '',
    # merge
    ConcatLayer: '',
    ElemwiseMergeLayer: '',
    ElemwiseSumLayer: '',
    # normalization
    BatchNormLayer: '',
    LocalResponseNormalization2DLayer: '',
    # embedding
    EmbeddingLayer: '',
    # special
    NonlinearityLayer: '',
    BiasLayer: '',
    ScaleLayer: '',
    ExpressionLayer: '',
    InverseLayer: '',
    TransformerLayer: '',
    TPSTransformerLayer: '',
    ParametricRectifierLayer: '',
    RandomizedRectifierLayer: '',
}

VERBOSE = {
    # input
    InputLayer: 'input: {output_shape}',
    # dense
    DenseLayer: '''fully connected
    W: {W:param}
    bias: {b:param}
    nonlinearity: {nonlinearity:func}
    output: {output_shape}''',
    NINLayer: '''network in network
    units: {num_units}
    W: {W:param}
    bias: {b:param}
    nonlinearity: {nonlinearity:func}
    output: {output_shape}
    ''',
    # convolution
    Conv1DLayer: '''convolution
    filters: {num_filters}
    filter size: {filter_size}
    convolution: {convolution:func}
    weights: {W:param}
    bias: {b:param}
    stride: {stride:shape}
    padding: {pad}
    nonlinearity: {nonlinearity:func}
    output: {output_shape}''',
    Conv2DLayer: '''convolution
    filters: {num_filters}
    filter size: {filter_size:shape}
    convolution: {convolution:func}
    weights: {W:param}
    bias: {b:param}
    stride: {stride:shape}
    padding: {pad}
    nonlinearity: {nonlinearity:func}
    output: {output_shape}''',
    Conv3DLayer: '''convolution
    filters: {num_filters}
    filter size: {filter_size}
    convolution: {convolution:func}
    weights: {W:param}
    bias: {b:param}
    stride: {stride:shape}
    padding: {pad}
    nonlinearity: {nonlinearity:func}
    output: {output_shape}''',
    TransposedConv2DLayer: '''de-convolution,
    filters: {num_filters}
    filter size: {filter_size}
    weights: {W:param}
    bias: {b:param}
    stride: {stride:shape}
    cropping: {crop}
    nonlinearity: {nonlinearity:func}
    output: {output_shape}''',
    TransposedConv3DLayer: '''de-convolution,
    filters: {num_filters}
    filter size: {filter_size}
    weights: {W:param}
    bias: {b:param}
    stride: {stride:shape}
    cropping: {crop}
    nonlinearity: {nonlinearity:func}
    output: {output_shape}''',
    Deconv2DLayer: '''de-convolution
    filters: {num_filters}
    filter size: {filter_size}
    weights: {W:param}
    bias: {b:param}
    stride: {stride:shape}
    cropping: {crop}
    nonlinearity: {nonlinearity:func}
    output: {output_shape}''',
    DilatedConv2DLayer: '''dilated conv.
    filters: {num_filters}
    filter size: {filter_size}
    dilation: {dilation}
    weights: {W:param}
    bias: {b:param}
    padding: {pad}
    nonlinearity: {nonlinearity:func}
    output: {output_shape}''',
    LocallyConnected2DLayer: '''
    filters: {num_filters}
    filter size: {filter_size}
    weights: {W:param}
    bias: {b:param}
    stride: {stride:shape}
    padding: {pad}
    nonlinearity: {nonlinearity:func}
    channel wise : {channelwise}
    output: {output_shape}''',
    # pooling
    Pool1DLayer: '''pooling
    pool size: {pool_size:shape}
    stride: {stride:shape}
    pad: {pad:list}
    mode: {mode}
    output: {output_shape}''',
    Pool2DLayer: '''pooling
    pool size : {pool_size:shape}
    stride: {stride:shape}
    pad: {pad:list}
    mode: {mode}
    output: {output_shape}''',
    Pool3DLayer: '''pooling
    pool size: {pool_size:shape}
    stride: {stride:shape}
    pad: {pad:list}
    mode: {mode}
    output: {output_shape}''',
    MaxPool1DLayer: '''max-pooling
    pool size: {pool_size:shape}
    stride: {stride:shape}
    pad: {pad:list}
    output: {output_shape}''',
    MaxPool2DLayer: '''max-pooling
    pool size: {pool_size:shape}
    stride: {stride:shape}
    pad: {pad:list}
    output: {output_shape}''',
    MaxPool3DLayer: '''max-pooling
    pool size: {pool_size:shape}
    stride: {stride:shape}
    pad: {pad:list}
    output: {output_shape}''',
    Upscale1DLayer: '''upscale
    scale factor: {scale_factor:list}
    output: {output_shape}''',
    Upscale2DLayer: '''upscale
    scale factor: {scale_factor:list}
    output: {output_shape}''',
    Upscale3DLayer: '''upscale
    scale factor: {scale_factor:list}
    output: {output_shape}''',
    GlobalPoolLayer: '''global pooling
    function: {pool_function:func}
    output: {output_shape}''',
    FeaturePoolLayer: '''feature pooling
    function: {pool_function:func}
    pool size: {pool_size:shape}
    axis: {axis}
    output: {output_shape}''',
    FeatureWTALayer: '''WTA feat. pool.
    pool size: {pool_size:shape}
    axis: {axis}
    output: {output_shape}''',
    SpatialPyramidPoolingLayer: '''pyramid pooling
    pool. dimentions: {pool_dims}
    mode: {mode}
    implementation: {implementation}
    output: {output_shape}''',
    # recurrent
    CustomRecurrentLayer: '',
    RecurrentLayer: '',
    LSTMLayer: '',
    GRULayer: '',
    Gate: '',
    # noise
    DropoutLayer: '''dropout
    dropout prob.: {p:0.3%}
    rescale outputs: {rescale}
    shared axes: {shared_axes}''',
    GaussianNoiseLayer: '''gauss. noise
    std. deviation: {sigma}''',
    # shape
    ReshapeLayer: '''reshape
    output: {output_shape}''',
    FlattenLayer: '''flatten
    output dims.: {outdim}
    output: {output_shape}''',
    DimshuffleLayer: '''dim-shuffle
    pattern: {pattern}
    output: {output_shape}''',
    PadLayer: '''padding
    value: {val}
    width: {width:list}
    since dim: {batch_ndim}
    output: {output_shape}''',
    SliceLayer: '''slicing
    indices: {indices}
    axis: {axis}
    output: {output_shape}''',
    # merge
    ConcatLayer: '''concatenation
    axis: {axis}
    cropping: {cropping:list}
    output: {output_shape}''',
    ElemwiseMergeLayer: '''elem-wise merge
    function: {merge_function:func}
    cropping: {cropping:list}
    output: {output_shape}''',
    ElemwiseSumLayer: '''elem-wise sum
    coefficients: {coeffs:list}
    cropping: {cropping:list}
    output: {output_shape}''',
    # normalization
    LocalResponseNormalization2DLayer: '''LRN
    alpha: {alpha:value}
    k: {k:value}
    beta: {beta:value}
    n: {n}''',
    BatchNormLayer: '''batch normalization
    alpha: {alpha:value}
    beta: {beta:param}
    epsilon: {epsilon:value}
    gamma: {gamma:param}
    axes: {axes:list}''',
    # embedding
    EmbeddingLayer: '''embedding
    input size: {input_size}
    output size: {output_size}
    output: {output_shape}''',
    # special
    NonlinearityLayer: 'nonlinearity: {nonlinearity:func}',
    BiasLayer: '''bias
    bias: {b:param}
    shared axes: {shared_axes:list}''',
    ScaleLayer: '''scaling
    scales: {scales:param}
    shared axes: {shared_axes:list}''',
    ExpressionLayer: '''expression
    function: {function:func}
    output: {output_shape}''',
    InverseLayer: '''inverse
    layer: {layer}
    output: {output_shape}''',
    TransformerLayer: '''
    network: {localization_network}
    downsample factor: {downsample_factor:list}
    border mode: {border_mode}
    output: {output_shape}''',
    TPSTransformerLayer: '''spacial trans.
    network: {localization_network}
    downsample factor: {downsample_factor:list}
    control points: {control_points}
    precompute grid: {precompute_grid}
    border mode: {border_mode}
    output: {output_shape}''',
    ParametricRectifierLayer: '''PReLU
    alpha: {alpha:value}
    shared axes: {shared_axes}''',
    RandomizedRectifierLayer: '''RReLU
    lower bound: {lower:value}
    upper bound: {upper:value}
    shared axes: {shared_axes}''',
}

SHORT = {
    # input
    InputLayer: 'input: {output_shape}',
    # dense
    DenseLayer: '''fully connected
    nonlinearity: {nonlinearity:func}
    output: {output_shape}''',
    NINLayer: '''NiN
    nonlinearity: {nonlinearity:func}
    output: {output_shape}''',
    # convolution
    Conv1DLayer: '''conv. {num_filters}, {filter_size:shape} \\\\{stride:shape}
    output: {output_shape}''',
    Conv2DLayer: '''conv. {num_filters}, {filter_size:shape} \\\\{stride:shape}
    output: {output_shape}''',
    Conv3DLayer: '''conv. {num_filters}, {filter_size:shape} \\\\{stride:shape}
    output: {output_shape}''',
    TransposedConv2DLayer:
    '''de-conv. {num_filters}, {filter_size:shape} \\\\{stride:shape}
    output: {output_shape}''',
    TransposedConv3DLayer:
    '''de-conv. {num_filters}, {filter_size:shape} \\\\{stride:shape}
    output: {output_shape}''',
    Deconv2DLayer:
    '''de-conv. {num_filters}, {filter_size:shape} \\\\{stride:shape}
    output: {output_shape}''',
    DilatedConv2DLayer: '',
    # local
    LocallyConnected2DLayer: '',
    # pooling
    Pool1DLayer: '''pool. {pool_size:shape} \\\\{stride:shape}
    output: {output_shape}''',
    Pool2DLayer: '''pool. {pool_size:shape} \\\\{stride:shape}
    output: {output_shape}''',
    Pool3DLayer: '''pool. {pool_size:shape} \\\\{stride:shape}
    output: {output_shape}''',
    MaxPool1DLayer: '''max-pool. {pool_size:shape} \\\\{stride:shape}
    output: {output_shape}''',
    MaxPool2DLayer: '''max-pool. {pool_size:shape} \\\\{stride:shape}
    output: {output_shape}''',
    MaxPool3DLayer: '''max-pool. {pool_size:shape} \\\\{stride:shape}
    output: {output_shape}''',
    Upscale1DLayer: '''upscale {scale_factor:list}
    output: {output_shape}''',
    Upscale2DLayer: '''upscale {scale_factor:list}
    output: {output_shape}''',
    Upscale3DLayer: '''upscale {scale_factor:list}
    output: {output_shape}''',
    GlobalPoolLayer: '''global pooling: {pool_function:func}
    output: {output_shape}''',
    FeaturePoolLayer: '',
    FeatureWTALayer: '',
    SpatialPyramidPoolingLayer: '',
    # recurrent
    CustomRecurrentLayer: '',
    RecurrentLayer: '',
    LSTMLayer: '',
    GRULayer: '',
    Gate: '',
    # noise
    DropoutLayer: 'dropout, {p:0.2%}',
    GaussianNoiseLayer: 'noise (sigma:value)',
    # shape
    ReshapeLayer: 'reshape\noutput: {output_shape}',
    FlattenLayer: 'flatten\noutput: {output_shape}',
    DimshuffleLayer: 'dim-shuffle ({pattern})\noutput: {output_shape}',
    PadLayer: '''padding ({val} \\{width})
    output: {output_shape}''',
    SliceLayer: '',
    # merge
    ConcatLayer: 'concatenation\noutput: {output_shape}',
    ElemwiseMergeLayer: 'merge, {merge_function:func}',
    ElemwiseSumLayer: '+',
    # normalization
    BatchNormLayer: 'BN',
    LocalResponseNormalization2DLayer: '',
    # embedding
    EmbeddingLayer: '',
    # special
    NonlinearityLayer: '{nonlinearity:func}',
    BiasLayer: 'bias',
    ScaleLayer: 'scaling',
    ExpressionLayer: 'expression\noutput: {output_shape}',
    InverseLayer: '',
    TransformerLayer: '',
    TPSTransformerLayer: '',
    ParametricRectifierLayer: 'PReLU',
    RandomizedRectifierLayer: 'RReLU',
}


class ParamFormatter(Formatter):
    """A special :class:`Formatter` for the layer attributes.

    The formatter will (somewhat) nicely format the input and output
    shapes, they are formatted and also offers special format specs
    for formatting the parameters of a layer.

    The format specs are:
    - ``'shape'``: to format a shape like ``'32x32'``
    - ``'func'``: to show the name of a function.
    - ``'param'``: will show the shape of the parameter in a list.
    - ``'list'``: will show the parameter as a list.
    - ``'value'``: to display the value of a parameter.
    """

    shapes = ('output_shape', 'input_shape')

    @staticmethod
    def activation_shape(shape):
        """Format a input and output shape."""
        if len(shape) == 2:
            return f'{shape[1]} units'
        elif len(shape) == 4:
            return '{} ch, {} x {}'.format(*shape[1:])
        elif len(shape) == 5:
            return '{} ch, {} x {} x {}'.format(*shape[1:])
        else:
            raise ValueError(f'Can not handle shape "{shape}".')

    @staticmethod
    def param_shape(shape):
        """Format a parameter shape."""
        if shape is None:
            return 'none'
        if len(shape) == 1:
            return f'[{shape[0]}, ]'
        return '[{}]'.format(', '.join(str(i) for i in shape))

    def get_value(self, key, args, kwargs):
        if not isinstance(key, str):
            return super(ParamFormatter, self).get_value(key, args, kwargs)

        if key in self.shapes:
            return self.activation_shape(kwargs[key])
        return super(ParamFormatter, self).get_value(key, args, kwargs)

    def format_field(self, value, format_spec):
        if format_spec == 'shape':
            return 'x'.join(str(i) for i in value)
        elif format_spec == 'func':
            try:
                return value.__name__
            except AttributeError:
                return value.__class__.__name__
        elif format_spec == 'param':
            if value is None:
                return 'none'
            shape = value.shape.eval()
            return self.param_shape(shape)
        elif format_spec == 'list':
            return self.param_shape(value)
        elif format_spec == 'value':
            try:
                value = value.eval()
            except AttributeError:
                pass
            return str(value)
        return super(ParamFormatter, self).format_field(value, format_spec)


def format_create(layer, format_map, color_map=DEFAULT_MAP,
                  formatter=ParamFormatter(), **kwargs):
    """Create a :class:`Node` from a formatting system.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The layer.
    format_map : a dictionary mapping layer types to format strings
        A dictionary that contains a format string for each of the
        layer's types. The information for the node is created by using
        ``formatter`` to call format with all the layer attributes as
        (keyword) arguments.
    color_map: a dictionary mapping layer types to strings
        The dictionary should contain all the colors for all the used
        layer types.
    formatter : :class:`Formatter` instance
        The formatter for creating the node information.
    kwargs : keyword arguments
        Those will be passed down to :class:`Node`.
    """
    color = color_map[type(layer)]
    variables = {n: getattr(layer, n) for n in dir(layer)}
    label = formatter.format(format_map[type(layer)], **variables)
    return Node(repr(layer), label=label, shape='record',
                fillcolor=color, style='filled', **kwargs)


def default_create(layer, verbose=False, **kwargs):
    """Default creation function for nodes.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The layer.
    verbose : boolean (``False``)
        Show extra information if ``True``.
    kwargs : keyword arguments
        Those will be passed to ``format_create`` and :class:`Node`.
    """
    frmt_dct = VERBOSE if verbose else SHORT
    return format_create(layer, frmt_dct, **kwargs)


# creating and drawing graphs

def draw_to_file(layer_or_layers, filename, node_creator=default_create,
                 **kwargs):
    """Draws a network diagram to a file.

    Parameters
    ----------
    layer_or_layers : one :class:`Layer` instance or a list of layers
        Either a list of layers or the model in form of the last layer.
    filename : string
        The filename to save the output to.
    node_creator : callable
        A function that creates a :class:`Node` for a given layer.
    kwargs : keyword arguments
        Those will be passed to ``pydot_graph``, ``node_creator`` and
        later to :class:`Node`.
    """
    if isinstance(layer_or_layers, Layer):
        layers = get_all_layers(layer_or_layers)
    else:
        layers = layer_or_layers
    dot = pydot_graph(layers, node_creator=node_creator, **kwargs)
    ext = filename.rsplit('.', 1)[1]
    with open(filename, 'wb') as fid:
        fid.write(dot.create(format=ext))


def draw_to_notebook(layer_or_layers, node_creator=default_create, **kwargs):
    """Draws a network diagram in an IPython notebook.

    Parameters
    ----------
    layer_or_layers : one :class:`Layer` instance or a list of layers
        Either a list of layers or the model in form of the last layer.
    node_creator : callable
        A function that creates a :class:`Node` for a given layer.
    kwargs : keyword arguments
        Those will be passed to ``pydot_graph``, ``node_creator`` and
        later to :class:`Node`.
    """
    from IPython.display import Image
    if isinstance(layer_or_layers, Layer):
        layers = get_all_layers(layer_or_layers)
    else:
        layers = layer_or_layers
    dot = pydot_graph(layers, node_creator=node_creator, **kwargs)
    return Image(dot.create_png())


def pydot_graph(layers, node_creator=default_create, **kwargs):
    """Create a :class:`Dot` graph for a list of layers

    Parameters
    ----------
    layers : list of :class:`Layer` instances
        The graph will be created with the layers from that list.
    node_creator : callable (``default_create``)
        A function that creates a :class:`Node` for a given layer.
    kwargs : keyword arguments
        Those will be passed down to ``node_creator`` or :class:`Node`.
    """
    nodes = {}
    edges = []
    for layer in layers:
        nodes[layer] = node_creator(layer, **kwargs)
        if hasattr(layer, 'input_layers'):
            for input_layer in layer.input_layers:
                edges.append((input_layer, layer))
        if hasattr(layer, 'input_layer'):
            edges.append((layer.input_layer, layer))

    graph = Dot('Network', graph_type='digraph')
    for node in nodes.values():
        graph.add_node(node)
    for start, end in edges:
        try:
            graph.add_edge(Edge(nodes[start], nodes[end]))
        except KeyError:
            pass
    return graph
