"""The CIFAR-10 and CIFAR-100 data sets with image augmentation.

This module provides the CIFAR-10 and CIFAR-100 data sets with on-the-fly
image augmentation as done in the `Deeply-Supervised Nets
<https://arxiv.org/abs/1409.5185>`_ and other papers.
"""
import numpy
from lasagne.utils import floatX

from . import DataSet, TrainingData, cifar


__all__ = ('CIFAR10', 'CIFAR100')


class Augmentation(object):
    """Wrapper class for the on-the-fly image augmentation.

    The data is zero padded four pixels at each side and then (on-the-fly)
    cropped to the 32x32. After that the image is randomly flipped
    horizontally.

    Parameters
    ----------
    data : ``numpy.array``
        The data to augment.
    random_state : ``numpy.random.RandomState``
        The random state used for the image augmentation.
    """

    def __init__(self, data, random_state):
        self.random = random_state
        self.shape = data.shape
        # zero padding 4 pixels left, right, top, bottom
        data = numpy.pad(data, ((0, 0), (0, 0), (4, 4), (4, 4)),
                         mode='constant')

        self.normal = data
        self.flipped = data[:, :, :, ::-1]

    @staticmethod
    def crop_image(image, x_offset, y_offset):
        """Crop a image with the given offsets."""
        return image[:, x_offset:x_offset + 32, y_offset:y_offset + 32]

    def __len__(self):
        return len(self.normal)

    def __getitem__(self, given):
        # TODO : better implementation
        normal = self.normal[given]
        flipped = self.flipped[given]

        if isinstance(given, int):
            x_off, y_off = self.random.randint(0, 8, size=(2, ))
            coin = self.random.choice((True, False))
            return self.crop_image(normal if coin else flipped, x_off, y_off)

        length = len(normal)
        coins = self.random.choice((True, False), size=(length, ))
        crops = self.random.randint(0, 8, size=(length, 2))
        iterator = zip(normal, flipped, coins, crops)
        return numpy.stack(self.crop_image(n if c else f, x, y)
                           for n, f, c, (x, y) in iterator)


class AugmentedTrainingData(TrainingData):
    """Perform pre-processing an image augmentation.

    This class performs the pre-processing and an on-the-fly image
    augmentation to the data.
    The pre-processing consist of global contrast normalization.
    To normalize other data use the ``normalize``-method.
    The data is also zero padded four pixels at each side and then
    (on-the-fly) cropped to the 32x32. After that the image may
    also be flipped horizontally.

    Parameters
    ----------
    data : ``numpy.array``
        The input data to augment.
    labels : ``numpy.array`` or ``list``
        The label information.
    random_state : dictionary or ``None`` (``None``)
        The random state to use for the training iterations and image
        augmentation.
    scale : float (``1.0``)
        The scale for the data.
    epsilon : float (``1e-8``)
       A small number added to the variance before normalizing the data,
       to avoid numerical problems.
    ddof : integer (``1``)
        The "Delta Degrees of Freedom" used for computing the variance.
    """

    def __init__(self, *args, scale=1.0, epsilon=1e-8, ddof=1, **kwargs):
        super(AugmentedTrainingData, self).__init__(*args, **kwargs)

        self.scale = floatX(scale)
        self.mean = numpy.mean(self.data, axis=(0, 2, 3), keepdims=True)[0]
        self.std = floatX(numpy.maximum(numpy.std(self.data, axis=(0, 2, 3),
                                                  keepdims=True, ddof=ddof)[0],
                                        epsilon))
        self._data = Augmentation(self.normalized(self.data), self.random)

    @property
    def state(self):
        return {'data': self.data.normal[:, :, 4:36, 4:36],
                'labels': self.labels, 'random_state': self.random.get_state(),
                'mean': self.mean, 'std': self.std, 'scale': self.scale}

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.mean = floatX(state['mean'])
        obj.std = floatX(state['std'])
        obj.scale = floatX(state['scale'])
        obj.random = numpy.random.RandomState()
        obj.random.set_state(state['random_state'])
        obj._labels = numpy.array(state['labels'], dtype=numpy.int32)
        obj._data = Augmentation(floatX(state['data']), obj.random)
        return obj

    def normalized(self, data):
        """Return a normalized version of the data."""
        return (data - self.mean) / self.std * self.scale


class AugmentedCIFAR(DataSet):
    """Perform pre-processing and image augmentation on a data set.

    This class performs pre-processing to the training, test and
    validation images and also performs data augmentation to the
    training set.
    The pre-processing consist of global contrast normalization.
    To normalize other data use the ``normalize``-method.
    The training data is also zero padded four pixels at each side and
    then (on-the-fly) cropped to the original size of 32x32. After that
    the image may also be flipped horizontally.

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
    root : string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite : boolean (``False``)
        If ``True`` any existing data will be overwritten.

    Note: The ``interval`` parameter is removed due to the global
    contrast normalization.
    """

    training_class = AugmentedTrainingData

    def __init__(self, *args, interval=None, **kwargs):
        if interval is not None:
            raise ValueError(
                'Augmentation does not support setting an interval.')
        super(AugmentedCIFAR, self).__init__(*args, interval=None, **kwargs)

        def normed(dataset):
            data, labels = dataset.set
            return self.test_class(self.training.normalized(data), labels)

        if self.test is None:
            self.validation = normed(self.validation)
        elif self.validation is self.test:
            self.test = self.validation = normed(self.validation)
        else:
            self.validation = normed(self.validation)
            self.test = normed(self.test)

    @property
    def pixel_mean(self):
        """The pixel mean of the training set."""
        return self.training.mean

    def normalized(self, data):
        """Normalize the given data."""
        return self.training.normalized(data)


class CIFAR10(AugmentedCIFAR, cifar.CIFAR10):
    """The CIFAR-10 data set.

    The CIFAR-10 data set contains 60,000 32x32 color images. The
    training set consist of 50,000 images the test set 10,000.
    The are 10 classes with overall 6,000 images per class. The
    classes have the labels: 'airplane', 'automobile', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship' and 'truck'.

    All the images are pre-processed by global contrast normalization.
    Additionally the training images are zero padded and randomly
    cropped to 32x32. The the normal or flipped version of the training
    image is returned. The image augmentation is following
    `Deeply-Supervised Nets <https://arxiv.org/abs/1409.5185>`_ and
    other papers.

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
    root : string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite : boolean (``False``)
        If ``True`` any existing data will be overwritten.

    Note: The ``interval`` parameter is removed due to the global
    contrast normalization.

    For more information on the CIFAR-10 data set see the `web site
    <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
    """

    pass


class CIFAR100(AugmentedCIFAR, cifar.CIFAR100):
    """The CIFAR-100 data set.

    As the CIFAR-10, the CIFAR-100 data set contains 60,000 32x32 color
    images. The training set consist of 50,000 images the test set
    10,000. The difference is that there are 100 classes with 600
    images per class.
    The 100 classes can be grouped into 20 super classes with 5
    sub classes each. The super classes are called coarse labels, the
    sub-classes are called to as fine labels.
    The label classes and super classes are:

    | super class                    | classes                         |
    |--------------------------------+---------------------------------|
    | aquatic mammals                | beaver, dolphin, otter, seal,   |
    |                                | whale                           |
    | fish                           | aquarium fish, flatfish, ray,   |
    |                                | shark, trout                    |
    | flowers                        | orchids, poppies, roses,        |
    |                                | sunflowers, tulips              |
    | food containers                | bottles, bowls, cans, cups,     |
    |                                | plates                          |
    | fruit and vegetables           | apples, mushrooms, oranges,     |
    |                                | pears, sweet peppers            |
    | household electrical devices   | clock, computer keyboard, lamp, |
    |                                | telephone, television           |
    | household furniture            | bed, chair, couch, table,       |
    |                                | wardrobe                        |
    | insects                        | bee, beetle, butterfly,         |
    |                                | caterpillar, cockroach          |
    | large carnivores               | bear, leopard, lion, tiger, wolf|
    | large man-made outdoor things  | bridge, castle, house, road,    |
    |                                | skyscraper                      |
    | large natural outdoor scenes   | cloud, forest, mountain, plain, |
    |                                | sea                             |
    | large omnivores and herbivores | camel, cattle, chimpanzee,      |
    |                                | elephant, kangaroo              |
    | medium-sized mammals           | fox, porcupine, possum, raccoon,|
    |                                | skunk                           |
    | non-insect invertebrates       | crab, lobster, snail, spider,   |
    |                                | worm                            |
    | people                         | baby, boy, girl, man, woman     |
    | reptiles                       | crocodile, dinosaur, lizard,    |
    |                                | snake, turtle                   |
    | small mammals                  | hamster, mouse, rabbit, shrew,  |
    |                                | squirrel                        |
    | trees                          | maple, oak, palm, pine, willow  |
    | vehicles 1                     | bicycle, bus, motorcycle,       |
    |                                | pickup truck, train             |
    | vehicles 2                     | lawn-mower, rocket, streetcar,  |
    |                                | tank, tractor                   |

    All the images are pre-processed by global contrast normalization.
    Additionally the training images are zero padded and randomly
    cropped to 32x32. The the normal or flipped version of the training
    image is returned. The image augmentation is following
    `Deeply-Supervised Nets <https://arxiv.org/abs/1409.5185>`_ and
    other papers.

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
    root : string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite : boolean (``False``)
        If ``True`` any existing data will be overwritten.

    Note: The ``interval`` parameter is removed due to the global
    contrast normalization.

    For more information on the CIFAR-10 data set see the `web site
    <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
    """

    pass
