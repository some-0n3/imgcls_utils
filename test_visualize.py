from itertools import product

from lasagne.layers.base import Layer, MergeLayer
from parameterized import param, parameterized
from pytest import skip

from . import visualize


MAPS = ('SHORT', 'VERBOSE', 'EMPTY', 'DEFAULT_MAP')
LAYERS = visualize._types_from_lasange()


def name_func(testcase_func, _, param):
    """Create a name for the test function."""
    return '{}_{}_{}'.format(testcase_func.__name__, param.args[0],
                             param.args[1].__name__)


@parameterized.expand([param(k, t) for k, t in product(MAPS, LAYERS)],
                      testcase_func_name=name_func)
def test_key(map_name, layer):
    if layer is Layer or layer is MergeLayer:
        skip()
    assert layer in getattr(visualize, map_name)


@parameterized.expand([param(k, t) for k, t in product(MAPS, LAYERS)],
                      testcase_func_name=name_func)
def test_entry(map_name, layer):
    if layer is Layer or layer is MergeLayer or map_name == 'EMPTY':
        skip()
    dct = getattr(visualize, map_name)
    if layer not in dct:
        skip()
    assert dct[layer]
