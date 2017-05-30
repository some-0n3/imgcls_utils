"""Create a distractor image set from MNIST."""
import numpy
from theano import config

from .mnist import MNIST


__all__ = ('Distractor', )


def patch_img(*patches):
    """Patch multiple images together (besides each other)."""
    return numpy.concatenate(patches, axis=-1)


def fix_labels(labels, targets=(0, 1, 2, 3)):
    """Convert label information into target / not target."""
    return numpy.vectorize(targets.__contains__)(labels).astype(numpy.int32)


class Augmentation(object):
    """
    Create a distractor data set.

    This class takes a number of images and labels. The labels are
    either ``1`` if the image shows an object of the target class,
    or ``0`` if it shows a distractor.
    Every time an image is requested a distractor is randomly added
    to either side of the image.

    Parameters
    ----------
    data: ``numpy.array``
        The image data.
    labels: ``numpy.array``
        The label information.
    """

    def __init__(self, data, labels):
        self.data = data
        self.distractors = data[numpy.nonzero(1 - labels)[0]]
        *dims, weight = data.shape
        self.shape = tuple((*dims, weight * 2))
        self.dims = self.shape[1:]

    def __len__(self):
        return len(self.data)

    def get_distractors(self, size=None):
        """Return one or more random distractors."""
        idx = numpy.random.randint(0, len(self.distractors), size=size)
        return self.distractors[idx]

    def __getitem__(self, given):
        data = self.data[given]

        if isinstance(given, int):
            coin = numpy.random.choice((True, False))
            distr = self.get_distractors()
            return patch_img(data, distr) if coin else patch_img(distr, data)

        length = len(data)
        batch = numpy.empty((length, *self.dims), dtype=config.floatX)
        coins = numpy.random.choice((True, False), size=(length, ))
        distr = self.get_distractors(size=(length, ))
        for i, (img, coin, dis) in enumerate(zip(data, coins, distr)):
            batch[i] = patch_img(img, dis) if coin else patch_img(dis, img)
        return batch


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
    testsplit : float in [0, 1] or integer (``0``)
        Says how may data points from the training set are reserved
        for the test set. If ``testsplit`` is ``0`` (or less) the
        validation set is used as test set.
        The parameter is either a float in [0, 1] describing the split
        in percent or an integer describing the number of elements.
    interval : tuple (pair) of numbers or ``None`` (``None``)
        The interval to put the data points into.
    root : string (``'./_datasets'``)
        The root-directory for all the files (downloaded and cached).
    overwrite : boolean (``False``)
        If ``True`` any existing data will be overwritten.
    """

    cache_file = 'mnist_distractor.pkl'
    targets = (0, 1, 2, 3)

    def __init__(self, *args, **kwargs):
        super(Distractor, self).__init__(*args, **kwargs)
        self._x_train = Augmentation(self._x_train, self._y_train)
        self._x_valid = Augmentation(self._x_valid, self._y_valid)
        self._x_test = Augmentation(self._x_test, self._y_test)

    def create(self, root='./_datasets'):
        dct = super(Distractor, self).create(root=root)
        dct['training labels'] = fix_labels(dct['training labels'],
                                            targets=self.targets)
        dct['validation labels'] = fix_labels(dct['validation labels'],
                                              targets=self.targets)
        return dct
