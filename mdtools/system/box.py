import numpy as np

from MDAnalysis.lib.util import check_box
from MDAnalysis.lib.mdamath import triclinic_vectors

from ..lib.distances import _min_image_triclinic, _wrap_triclinic

#==============================================================================#

class Box:
    """A periodic simulation box in three dimensions.

    Implements methods for wrapping coordinates, minimum image convention,
    and generating random positions in the box.

    Parameters
    ----------
    dims : array-like
        A three- or three-component array containing three box lengths,
        or three box lengths and three side angles
    """
    def __init__(self, dims):
        assert len(dims) == 3 or len(dims) == 6, "Invalid number of box dimensions"

        if len(dims) == 3:
            dims = [*dims, 90., 90., 90]
        self.dims = np.array(dims)
        self.h = triclinic_vectors(dims)
        self.hinv = np.linalg.inv(self.h)

        boxtype, box = check_box(self.dimensions)
        self.boxtype = boxtype

    def __repr__(self):
        return "Box(dims = [{:.3g}, {:.3g}, {:.3g}, {:.3g}, {:.3g}, {:.3g}])".format(*self.dimensions)

    @property
    def dimensions(self):
        return self.dims

    def is_orthogonal(self):
        return self.boxtype == "ortho"

    def volume(self):
        return np.linalg.det(self.h)

    def random_position(self):
        return np.dot(np.random.rand(3), self.h)

    def to_fractional(self, r):
        return np.dot(r, self.hinv)

    def to_real(self, f):
        return np.dot(f, self.h)

    def image_flag(self, r):
        return np.floor(np.dot(r, self.hinv)).astype('int')

    def min_image(self, r):
        _min_image_triclinic(r, self.h, self.hinv)

    def wrap(self, r, img = None):
        _wrap_triclinic(r, self.h, self.hinv, img)