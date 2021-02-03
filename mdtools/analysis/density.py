"""
A tool to compute mass and charge density profiles along the three
cartesian axes of the simulation cell. Works only for orthorombic,
fixed volume cells (thus for simulations in canonical NVT ensemble).
"""
from __future__ import division, absolute_import

import os.path as path

import numpy as np

from MDAnalysis.analysis.base import AnalysisBase

class LinearDensity(AnalysisBase):
    """
    Linear density profile.
    A nearly identical port from the raw MDAnalysis class.

    Parameters
    ----------
    select : AtomGroup
          any atomgroup
    grouping : str {'atoms', 'residues', 'segments', 'fragments'}
          Density profiles will be computed on the center of geometry
          of a selected group of atoms ['atoms']
    binsize : float
          Bin width in Angstrom used to build linear density
          histograms. Defines the resolution of the resulting density
          profile (smaller --> higher resolution) [0.25]
    verbose : bool (optional)
          Show detailed progress of the calculation if set to ``True``; the
          default is ``False``.

    Attributes
    ----------
    results : dict
          Keys 'x', 'y', and 'z' for the three directions. Under these
          keys, find 'pos', 'pos_std' (mass-weighted density and
          standard deviation), 'char', 'char_std' (charge density and
          its standard deviation), 'slice_volume' (volume of bin).

    Example
    -------
    First create a LinearDensity object by supplying a selection,
    then use the :meth:`run` method::

      ldens = LinearDensity(selection)
      ldens.run()


    .. versionadded:: 0.14.0

    .. versionchanged:: 1.0.0
       Support for the ``start``, ``stop``, and ``step`` keywords has been
       removed. These should instead be passed to :meth:`LinearDensity.run`.
       The ``save()`` method was also removed, you can use ``np.savetxt()`` or
       ``np.save()`` on the :attr:`LinearDensity.results` dictionary contents
       instead.

    .. versionchanged:: 1.0.0
       Changed `selection` keyword to `select`
    """

    def __init__(self, select, grouping = 'atoms', binsize = 0.25, **kwargs):
        super(LinearDensity, self).__init__(select.universe.trajectory,
                                            **kwargs)
        # allows use of run(parallel=True)
        self._ags = [select]
        self._universe = select.universe

        self.binsize = binsize

        # group of atoms on which to compute the COM (same as used in
        # AtomGroup.wrap())
        self.grouping = grouping

        # Dictionary containing results
        self.results = {'x': {'dim': 0}, 'y': {'dim': 1}, 'z': {'dim': 2}}
        # Box sides
        self.dimensions = self._universe.dimensions[:3]
        self.volume = np.prod(self.dimensions)
        # number of bins
        bins = (self.dimensions // self.binsize).astype(int)

        # Here we choose a number of bins of the largest cell side so that
        # x, y and z values can use the same "coord" column in the output file
        self.nbins = bins.max()
        slices_vol = self.volume / bins

        self.keys = ['mass', 'mass_std', 'charge', 'charge_std']

        # Initialize results array with zeros
        for dim in self.results:
            idx = self.results[dim]['dim']
            self.results[dim].update({'slice volume': slices_vol[idx]})
            for key in self.keys:
                self.results[dim].update({key: np.zeros(self.nbins)})

        # Variables later defined in _prepare() method
        self.masses = None
        self.charges = None
        self.totalmass = None

    def _prepare(self):
        # group must be a local variable, otherwise there will be
        # issues with parallelization
        group = getattr(self._ags[0], self.grouping)

        # Get masses and charges for the selection
        try:  # in case it's not an atom
            self.masses = np.array([elem.total_mass() for elem in group])
            self.charges = np.array([elem.total_charge() for elem in group])
        except AttributeError:  # much much faster for atoms
            self.masses = self._ags[0].masses
            self.charges = self._ags[0].charges

        self.totalmass = np.sum(self.masses)

        # Find position bins for the histogram by choosing first gorup
        _, edges = np.histogram([],bins=self.nbins, range=(0.0, max(self.dimensions)))
        self.results['bins'] = (edges[:-1] + edges[1:])/2.0

    def _single_frame(self):
        self.group = getattr(self._ags[0], self.grouping)
        self._ags[0].wrap(compound=self.grouping)

        # Find position of atom/group of atoms
        if self.grouping == 'atoms':
            positions = self._ags[0].positions  # faster for atoms
        else:
            # COM for res/frag/etc
            positions = np.array([elem.centroid() for elem in self.group])

        for dim in ['x', 'y', 'z']:
            idx = self.results[dim]['dim']

            key = 'mass'
            key_std = 'mass_std'
            # histogram for positions weighted on masses
            hist, _ = np.histogram(positions[:, idx],
                                   weights=self.masses,
                                   bins=self.nbins,
                                   range=(0.0, max(self.dimensions)))

            self.results[dim][key] += hist
            self.results[dim][key_std] += np.square(hist)

            key = 'charge'
            key_std = 'charge_std'
            # histogram for positions weighted on charges
            hist, _ = np.histogram(positions[:, idx],
                                   weights=self.charges,
                                   bins=self.nbins,
                                   range=(0.0, max(self.dimensions)))

            self.results[dim][key] += hist
            self.results[dim][key_std] += np.square(hist)

    def _conclude(self):
        # Average results over the  number of configurations
        for dim in ['x', 'y', 'z']:
            for key in ['mass', 'mass_std', 'charge', 'charge_std']:
                self.results[dim][key] /= self.n_frames
            # Compute standard deviation for the error
            self.results[dim]['mass_std'] = np.sqrt(self.results[dim][
                'mass_std'] - np.square(self.results[dim]['mass']))
            self.results[dim]['char_std'] = np.sqrt(self.results[dim][
                'charge_std'] - np.square(self.results[dim]['charge']))

        for dim in ['x', 'y', 'z']:
            self.results[dim]['mass'] /= self.results[dim]['slice volume'] 
            self.results[dim]['charge'] /= self.results[dim]['slice volume']
            self.results[dim]['mass_std'] /= self.results[dim]['slice volume']
            self.results[dim]['charge_std'] /= self.results[dim]['slice volume']

    def _add_other_results(self, other):
        # For parallel analysis
        results = self.results
        for dim in ['x', 'y', 'z']:
            key = 'mass'
            key_std = 'mass_std'
            results[dim][key] += other[dim][key]
            results[dim][key_std] += other[dim][key_std]

            key = 'charge'
            key_std = 'charge_std'
            results[dim][key] += other[dim][key]
            results[dim][key_std] += other[dim][key_std]
