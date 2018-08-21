"""Create a distractor image set from MNIST."""
import numpy
from lasagne.utils import floatX

from . import Data, TestData, TrainingData
from .mnist import MNIST


__all__ = ('Distractor', )


def patch_img(*patches):
    """Patch multiple images together (besides each other)."""
    return numpy.concatenate(patches, axis=-1)


def fix_labels(labels, targets=(0, 1, 2, 3)):
    """Convert label information into target / not target."""
    return numpy.vectorize(targets.__contains__)(labels)


class Augmentation():
    """Wrapper class for the on-the-fly image augmentation.

    This class takes a number of input images and distractors. Every
    time an image is requested a distractor is randomly added to either
    side of the image.

    Parameters
    ----------
    data : ``numpy.array``
        The image data to augment.
    distractors : ``numpy.array``
        The distractor image to use for augmentation.
    random_state : ``numpy.random.RandomState``
        The random state used for the image augmentation.
    """

    def __init__(self, data, distractors, random_state):
        self.data = data
        self.distractors = distractors
        *dims, weight = data.shape
        self.shape = tuple((*dims, weight * 2))
        self.dims = self.shape[1:]
        self.random = random_state

    def __len__(self):
        return len(self.data)

    def get_distractors(self, size=None):
        """Return one or more random distractors."""
        idx = self.random.randint(0, len(self.distractors), size=size)
        return self.distractors[idx]

    def __getitem__(self, given):
        data = self.data[given]

        if isinstance(given, int):
            coin = self.random.choice((True, False))
            distr = self.get_distractors()
            return patch_img(data, distr) if coin else patch_img(distr, data)

        length = len(data)
        batch = numpy.empty((length, *self.dims), dtype=data.dtype)
        coins = self.random.choice((True, False), size=(length, ))
        distr = self.get_distractors(size=(length, ))
        for i, (img, coin, dis) in enumerate(zip(data, coins, distr)):
            batch[i] = patch_img(img, dis) if coin else patch_img(dis, img)
        return batch


class AugmentedData(Data):
    """Perform the distractor image augmentation to a data set.

    This class adds a random distractor image to the input data.
    The distractor images are randomly chosen from all input images
    that are not of the target class (they are labeled with ``0``).
    A new distractor is added, to an image to either its left or right,
    every time the image is requested.

    Parameters
    ----------
    data : ``numpy.array``
        The input images.
    labels : ``numpy.array`` or list
        The label information.
    random_state : dictionary or ``None`` (``None``)
        The random state for the image augmentation.
    """

    def __init__(self, data, labels, random_state=None):
        self.random = numpy.random.RandomState()
        if random_state:
            self.random.set_state(random_state)

        data = floatX(data)
        distractors = data[numpy.nonzero(1 - labels)[0]]
        self._data = Augmentation(data, distractors, self.random)
        self._labels = numpy.array(labels, dtype=numpy.int32)

    @property
    def distractors(self):
        """All the distractor images used for augmentation."""
        return self._data.distractors

    @property
    def originals(self):
        """The original images, without distractors."""
        return self._data.data

    @property
    def state(self):
        return {'data': self.originals, 'labels': self.labels,
                'random_state': self.random.get_state()}


class Distractor(MNIST):
    """A distractor data set created from MNIST.

    The distractor data set is created by choosing a set of target
    digits, all other digit classes are considered distractors.
    At each image a randomly selected distractor is added at either
    the left or the right side of the image. The distractors are
    added only the fly and change every time you retrieve the image
    from the data set.

    As MNIST this data set consists of 70,000 monochrome images.
    The dimensions are however 56x28 instead of 28x28. The labels
    will tell whenever a image will show an object from the target
    class. They will will be ``1`` for the target class and ``0``
    otherwise.

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

    training_class = type('AugmentedTrainingData',
                          (AugmentedData, TrainingData), {})
    test_class = type('AugmentedTestData', (AugmentedData, TestData), {})
    cache_file = 'mnist_distractor.npz'
    targets = (0, 1, 2, 3)

    def extract(self, root='./_datasets'):
        dct = super(Distractor, self).extract(root=root)
        dct['training labels'] = fix_labels(dct['training labels'],
                                            targets=self.targets)
        dct['validation labels'] = fix_labels(dct['validation labels'],
                                              targets=self.targets)
        return dct
