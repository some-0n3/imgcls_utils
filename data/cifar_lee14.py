"""The CIFAR-10 and CIFAR-100 data sets with image augmentation.

This module provides the CIFAR-10 and CIFAR-100 data sets with on-the-fly
image augmentation as done in the `Deeply-Supervised Nets
<https://arxiv.org/abs/1409.5185>`_ and other papers.
"""
import numpy

from . import DataSet, cifar


__all__ = ('CIFAR10', 'CIFAR100')


class Augmentation(object):
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
        The data to augment.
    scale : float (``1.0``)
        The scale for the data.
    epsilon : float (``1e-8``)
       A small number added to the variance before normalizing the data,
       to avoid numerical problems.
    ddof : integer (``1``)
        The "Delta Degrees of Freedom" used for computing the variance.
    """

    def __init__(self, data, scale=1.0, epsilon=1e-8, ddof=1):
        self.shape = data.shape
        self.scale = scale

        self.mean = numpy.mean(data, axis=(0, 2, 3), keepdims=True)[0]
        data -= self.mean

        norm = numpy.var(data, axis=(0, 2, 3), keepdims=True, ddof=ddof)[0]
        self.normalizer = numpy.maximum(numpy.sqrt(norm), epsilon)

        data = data / self.normalizer * self.scale

        # zero padding 4 pixels left, right, top, bottom
        data = numpy.pad(data, ((0, 0), (0, 0), (4, 4), (4, 4)),
                         mode='constant')

        self.normal = data
        self.flipped = data[:, :, :, ::-1]

    def normalize(self, data):
        """Return a normalized version of the data."""
        return (data - self.mean) / self.normalizer * self.scale

    @staticmethod
    def crop_image(image, x_offset, y_offset):
        """Crop a image with the given offsets."""
        return image[:, x_offset:x_offset + 32, y_offset:y_offset + 32]

    def __len__(self):
        return len(self.normal)

    def __getitem__(self, given):
        normal = self.normal[given]
        flipped = self.flipped[given]

        if isinstance(given, int):
            x_off, y_off = numpy.random.randint(0, 8, size=(2, ))
            coin = numpy.random.choice((True, False))
            return self.crop_image(normal if coin else flipped, x_off, y_off)

        length = len(normal)
        coins = numpy.random.choice((True, False), size=(length, ))
        crops = numpy.random.randint(0, 8, size=(length, 2))
        iterator = zip(normal, flipped, coins, crops)
        return numpy.stack(self.crop_image(n if c else f, x, y)
                           for n, f, c, (x, y) in iterator)


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
    testsplit : float in [0, 1] or integer (``0``)
        Says how may data points from the training set are reserved
        for the test set. If ``testsplit`` is ``0`` (or less) the
        validation set is used as test set.
        The parameter is either a float in [0, 1] describing the split
        in percent or an integer describing the number of elements.
    root : string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite : boolean (``False``)
        If ``True`` any existing data will be overwritten.

    Note: The ``interval`` parameter is removed due to the global
    contrast normalization.
    """

    def __init__(self, *args, interval=None, **kwargs):
        if interval is not None:
            raise ValueError(
                'Augmentation does not support setting an interval.')
        super(AugmentedCIFAR, self).__init__(*args, interval=None, **kwargs)
        self._x_train = Augmentation(self._x_train)
        self._x_valid = self._x_train.normalize(self._x_valid)
        if self._x_test is not self._x_valid:
            self._x_test = self._x_train.normalize(self._x_test)

    @property
    def pixel_mean(self):
        """The pixel mean of the training set."""
        return self._x_train.mean

    def normalize(self, data):
        """Normalize the given data."""
        return self._x_train.normalize(data)


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
    testsplit : float in [0, 1] or integer (``0``)
        Says how may data points from the training set are reserved
        for the test set. If ``testsplit`` is ``0`` (or less) the
        validation set is used as test set.
        The parameter is either a float in [0, 1] describing the split
        in percent or an integer describing the number of elements.
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
    testsplit : float in [0, 1] or integer (``0``)
        Says how may data points from the training set are reserved
        for the test set. If ``testsplit`` is ``0`` (or less) the
        validation set is used as test set.
        The parameter is either a float in [0, 1] describing the split
        in percent or an integer describing the number of elements.
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
