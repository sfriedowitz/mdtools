import numpy as np
from MDAnalysis.analysis.base import AnalysisBase

#==============================================================================#

class SerialAnalysis(AnalysisBase):
    """
    Extension of the MDAnalysis `AnalysisBase` class 
    for defining multi frame analysis with serial iteration over the trajectory.
    """

    def __init__(self, trajectory, verbose = False, **kwargs):
        """
        Parameters
        ----------
        trajectory : mda.Reader
            A trajectory Reader
        verbose : bool, optional
           Turn on more logging and debugging, default ``False``
        """
        super().__init__(trajectory, verbose, **kwargs)

        self._verbose = verbose
        self.results = {}
    
    def save(self, prefix = "", **kwargs):
        """Save the gathered results to a file."""
        pass

class SingleGroupAnalysis(SerialAnalysis):
    """The base class for analysing a single AtomGroup only."""

    _allow_multiple_atomgroups = False

    def __init__(self, atomgroup, **kwargs):
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        self._atomgroup = atomgroup
        self._universe = atomgroup.universe

class MultiGroupAnalysis(SerialAnalysis):
    """The base class for analysing a single or multiple AtomGroups."""

    _allow_multiple_atomgroups = True

    def __init__(self, atomgroups, **kwargs):
        if type(atomgroups) not in [list, tuple, np.ndarray]:
            atomgroups = [atomgroups]
        
        # Check that all atomgroups are from same universe
        for ag in atomgroups[1:]:
            if ag.universe != atomgroups[0].universe:
                raise ValueError("Given AtomGroups are not from the same Universe.")

        super().__init__(atomgroups[0].universe.trajectory, **kwargs)
        self._atomgroups = atomgroups
        self._universe = atomgroups[0].universe