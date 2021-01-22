#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np

from .base import MultiGroupAnalysis
from ..lib.utils import repair_molecules, save_path

#==============================================================================#
# Timeseries analyzers
#==============================================================================#

class DipoleTrajectory(MultiGroupAnalysis):
    """
    Calculate the dipole and charge current trajectories for a series of atomgroups.
    All residues within a group must contain the same number of particles.
    The rotational dipole is calculated for each frame,
    while the current is calculated only for frames with velocity data.
    The translational dipole is only calculated if the ``nojump`` flag is specified.

    Parameters
    ----------
    *atomgroups : AtomGroup, multiple
        variable number of atom groups to be analyzed
    restypes : list
        list of residue types for each atom group in ("SP", "NM", "CM") (None)
    labels : list
        list of labels for each atom group (None)
    nojump : bool
        boolean flag for if trajectory is unwrapped and translational dipole can be computed (False)
    repair : bool
        boolean flag for if molecules should be repaired across periodic boundaries (False)

    Returns
    ----------
    results : dict
        dictionary containing the calculated timeseries for each atom group
    """

    def __init__(self, *args, restypes = None, labels = None, nojump = False, repair = False, **kwargs):
        super().__init__([*args], **kwargs)

        # Group types of each AG
        if restypes is not None:
            if len(restypes) != len(self._atomgroups):
                raise ValueError("Number of atomgroups and residue types not equal.")
        else:
            restypes = ["CM" for i in range(len(self._atomgroups))]
        self.restypes = restypes

        # Names of each AG
        if labels is not None:
            if len(labels) != len(self._atomgroups):
                raise ValueError("Number of atomgroups and label names not equal.")
        else:
            labels = ["ag{}".format(i) for i in range(len(self._atomgroups))]
        self.labels = labels

        self.nojump= nojump
        self.repair = repair

    def _prepare(self):
        self.nframes = np.ceil((self.stop - self.start) / self.step).astype(int)
        # Setup the results arrays
        self.results["dt"] = self._trajectory.dt * self.step
        self.results["time"] = np.round(self._trajectory.dt * np.arange(self.start, self.stop, self.step), decimals = 4)
        self.results["volume"] = np.zeros(self.nframes)

        for i, ag in enumerate(self._atomgroups):
            label = self.labels[i]
            self.results[label + "_MD"] = np.zeros((self.nframes, 3))
            self.results[label + "_MJ"] = np.zeros((self.nframes, 3))
            self.results[label + "_J"] = np.zeros((self.nframes, 3))

    def _single_frame(self):
        # Add volume volume
        self.results["volume"][self._frame_index] = self._ts.volume

        # Loop over each AG, gets a vector for P and J
        for i, ag in enumerate(self._atomgroups):

            # Repair if needed across boundaries, slow
            if self.repair:
                repair_molecules(ag)

            # Determines which calculation method to use for each AG
            label = self.labels[i]
            restype = self.restypes[i]

            # Base-case is zeros
            MD = np.zeros(3)
            MJ = np.zeros(3)
            J = np.zeros(3)

            if restype == "SP":
                # Single particle, only current and translational dipole exists
                if self.nojump:
                    MJ = np.dot(ag.charges, ag.positions)
                if self._ts.has_velocities:
                    J = np.dot(ag.charges, ag.velocities)

            elif restype == "NM":
                # Neutral molecule, only rotational dipole is non-zero
                MD = np.dot(ag.charges, ag.positions)

            else:
                # Generic calculation for charged molecules
                # Vectorized calculation for each residue in the group
                idx = np.argwhere(ag.residues.resids[:, np.newaxis] == ag.resids)
                idx = np.asarray(np.split(idx[:,1], np.cumsum(np.unique(idx[:, 0], return_counts=True)[1])[:-1]))

                pos = ag.positions[idx]
                ms = ag.masses[idx]
                qs = ag.charges[idx]

                mtot = np.sum(ms, axis = 1, keepdims = True)
                qtot = np.sum(qs, axis = 1, keepdims = True)

                rcm = np.sum(ms[:, :, np.newaxis] * pos, axis = 1) / mtot
                MD = np.sum(np.sum(qs[:, :, np.newaxis] * (pos - rcm[:, np.newaxis, :]), axis = 1), axis = 0)

                if self.nojump:
                   MJ = np.sum(qtot * rcm, axis = 0)

                if self._ts.has_velocities:
                    vel = ag.velocities[idx]
                    vcm = np.sum(ms[:, :, np.newaxis] * vel, axis = 1) / mtot
                    J = np.sum(qtot * vcm, axis = 0)
                 
            self.results[label + "_MD"][self._frame_index, :] = MD
            self.results[label + "_MJ"][self._frame_index, :] = MJ
            self.results[label + "_J"][self._frame_index, :] = J

    def _conclude(self):
        self.results["nframes"] = self._frame_index + 1

    def save(self, prefix = "", delimiter = " ", **kwargs):
        output = save_path(prefix)
        if self._verbose:
            print("Saving results to files at location: {}".format(output))

        # Create array and header depending on which items are present
        header = ["time", "volume"]
        data = [self.results["time"], self.results["volume"]]

        columns = ["MDX", "MDY", "MDZ", "MJX", "MJY", "MJZ", "JX", "JZ", "JY"]
        for label in labels:
            header.extend(["{}_{}".format(label, col) for col in colum])

            data.append(self.results[label + "_MD"])
            data.append(self.results[label + "_MJ"])
            data.append(self.results[label + "_J"])

        data = np.column_stack(data)
        np.savetxt(output + "dipole_trajectory.dat", data, delimiter = delimiter, header = delimiter.join(header))