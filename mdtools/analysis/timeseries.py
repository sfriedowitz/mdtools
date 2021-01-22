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
    Calculates the dipole and charge current trajectories for a series of atomgroups.
    All residues within a group must contain the same number of particles.

    Parameters
    ----------
    *atomgroups : AtomGroup, multiple
        variable number of atom groups to be analyzed
    restypes : list
        list of residue types for each atom group in ("SP", "NM", "CM") (None)
    labels : list
        list of labels for each atom group (None)
    current : bool
        boolean flag for if trajectory has velocity data and current can be computed (False)
    nojump : bool
        boolean flag for if trajectory is unwrapped and translational dipole can be computed (False)
    repair : bool
        boolean flag for if molecules should be repaired across periodic boundaries (False)

    Returns
    ----------
    results : dict
        dictionary containing the calculated timeseries for each atom group
    """

    def __init__(self, *args, restypes = None, labels = None, current = False, nojump = False, repair = False, **kwargs):
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

        self.current = current
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

            if self.current:
                self.results[label + "_J"] = np.zeros((self.nframes, 3))

            if self.nojump:
                self.results[label + "_MJ"] = np.zeros((self.nframes, 3))

    def _single_frame(self):
        # Add volume volume
        self.results["volume"][self._frame_index] = self._ts.volume

        has_vel = self._ts.has_velocities
        if self.current and not has_vel:
            raise RuntimeError("Cannot compute current for timestep {:d} with no velocity data!".format(self._ts.frame))

        # Loop over each AG, gets a vector for P and J
        for i, ag in enumerate(self._atomgroups):

            # Repair if needed across boundaries, slow
            if self.repair:
                repair_molecules(ag)

            # Determines which calculation method to use for each AG
            label = self.labels[i]
            restype = self.restypes[i]

            if restype == "SP":
                # Single particle
                # No center-of-mass dipole, only translational/velocity current if charged
                MD = np.zeros(3)

                if self.current:
                    J = np.dot(ag.charges, ag.velocities)

                if self.nojump:
                    MJ = np.dot(ag.charges, ag.positions)

            elif restype == "NM":
                # Neutral molecule
                # No dipole/velocity current, only rotational dipole
                MD = np.dot(ag.charges, ag.positions)

                if self.current:
                    J = np.zeros(3)

                if self.nojump:
                    MJ = np.zeros(3)

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

                if self.current:
                    vel = ag.velocities[idx]
                    vcm = np.sum(ms[:, :, np.newaxis] * vel, axis = 1) / mtot
                    J = np.sum(qtot * vcm, axis = 0)

                if self.nojump:
                   MJ = np.sum(qtot * rcm, axis = 0)
                 
            self.results[label + "_MD"][self._frame_index, :] = MD

            if self.current:
                self.results[label + "_J"][self._frame_index, :] = J

            if self.nojump:
                self.results[label + "_MJ"][self._frame_index, :] = MJ

    def _conclude(self):
        self.results["nframes"] = self._frame_index + 1

    def save(self, prefix = "", delimiter = " ", **kwargs):
        self.output = save_path(prefix)
        if self._verbose:
            print("Saving results to files at location: {}".format(self.output))

        # Create array and header depending on which items are present
        header = ["time", "vol"]
        data = [self.results["time"], self.results["volume"]]
        for i, label in enumerate(labels):
            if self.current and self.nojump:
                cols = ("MDX", "MDY", "MDZ", "MJX", "MJY", "MJZ", "JX", "JY", "JZ")
                header.extend(["{}_{}".format(label, col) for col in cols])

                data.append(self.results[label + "_MD"])
                data.append(self.results[label + "_MJ"])
                data.append(self.results[label + "_J"])

            elif self.current:
                cols = ("MDX", "MDY", "MDZ", "JX", "JY", "JZ")
                header.extend(["{}_{}".format(label, col) for col in cols])

                data.append(self.results[label + "_MD"])
                data.append(self.results[label + "_J"])

            elif self.nojump:
                cols = ("MDX", "MDY", "MDZ", "MJX", "MJY", "MJZ")
                header.extend(["{}_{}".format(label, col) for col in cols])

                data.append(self.results[label + "_MD"])
                data.append(self.results[label + "_MJ"])

        data = np.column_stack(data)
        np.savetxt(self.output + "dipole_trajectory.dat", data, delimiter = delimiter, header = delimiter.join(header))