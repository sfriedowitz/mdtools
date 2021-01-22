from __future__ import division
import numpy as np

from .chemistry import *

#==============================================================================#

class Species():
    """An abstract class representing a molecular species,
    composed of some monomer types, atoms, and bonded functionality.

    All concrete species derive from this parent class.
    A species implements methods for adding atoms and bonded units
    to a Topology graph.

    Parameters
    ----------
    id : integer
        Unique ID for this species
    monomers : array-like of MonomerType
        All MonomerType objects present in species
    natoms : integer
        Number of atoms in a given unit of this species
    """
    def __init__(self, id, monomers, natoms):
        self.id = id
        self.natoms = natoms
        self.monomers = monomers

    def generate_molecules(self, *args, **kwargs):
        """
        Add a given number of molecules of the species to a system topology.
        """
        pass

    def charge(self):
        """
        Return the total charge for a molecule of this species.
        """
        return 0.0

    def volume(self):
        """
        Return the total volume occupied by atoms in a molecule of this species.
        """
        return 1.0

class Point(Species):
    """
    Species representing a single coarse-grained point-like object.
    """
    def __init__(self, id, mon, **kwargs):
        super().__init__(id, [mon], 1)

    def charge(self):
        return self.monomers[0].charge

    def volume(self):
        return self.monomers[0].size**3

    def generate(self, nmol, mid0, topology, box):
        mon = self.monomers[0] # Only one monomer type for a point
        for mid in range(1, nmol+1):
            # Atom ID is set when adding to the topology
            ai = Atom(mon, mid = mid0 + mid, sid = self.id)
            ai.set_position(box.random_position())
            topology.add_atom(ai)

class Multiblock(Species):
    """
    Species representing a linear homopolymer with multiple distinct block types of monomers.
    """
    def __init__(self, id, block_mons, block_lens, **kwargs):
        if len(block_mons) != len(block_lens):
            raise ValueError("Number of monomers and blocks not equal.")

        mons = list(np.unique(block_mons))
        natoms = np.sum(block_lens)
        super().__init__(id, mons, natoms)

        self.block_mons = block_mons
        self.block_lens = np.array(block_lens)
        self.block_ids = np.array([mon.id for mon in block_mons])
        self.block_ends = np.cumsum(block_lens)
        self.block_starts = np.append([0], self.block_ends[:-1])
        self.nblocks = len(block_mons)

        self.bond_scale = kwargs.get("bond_scale", 1.25)
        self.bond_type = kwargs.get("bond_type", 1)

        # self.block_map = {}
        # for blk_id, mon in enumerate(block_mons):
        #     self.block_map[blk_id] = mon

    def charge(self):
        charge = 0.0
        for blk in range(self.nblocks):
            mon = self.blk2mon(blk)
            charge += self.block_lens[blk] * mon.charge
        return charge

    def volume(self):
        vol = 0.0
        for blk in range(self.nblocks):
            mon = self.blk2mon(blk)
            vol += self.block_lens[blk] * mon.size**3
        return vol

    def idx2mon(self, idx):
        assert 0 <= idx < self.natoms, "Monomer index greater than number of atoms in chain."
        for blk in range(self.nblocks):
            if idx < self.block_ends[blk]:
                return self.block_mons[blk]

    def blk2mon(self, blk):
        assert 0 <= blk < self.nblocks, "Block ID greater than number of blocks in chain."
        return self.block_mons[blk]

    def generate(self, nmol, mid0, topology, box):
        mon0 = self.blk2mon(0)
        for mid in range(1, nmol+1):
            # We place the first Atom randomly in the topology
            a0 = Atom(mon0, mid = mid0 + mid, sid = self.id)
            a0.set_position(box.random_position())
            topology.add_atom(a0)

            aprev = a0
            for ni in range(1, self.natoms):
                # Loop over and add all the connected atoms
                mon = self.idx2mon(ni)
                rbond = self.bond_scale*mon.size

                ai = Atom(mon, mid = mid0 + mid, sid = self.id)
                ai.set_image(aprev.img)

                # Add a random Gaussian displacement from previous
                delta = np.random.randn(3)
                delta *= rbond / np.linalg.norm(delta)
                ai.set_position(aprev.pos + delta)
                box.wrap(ai.pos, ai.img)

                topology.add_atom_bonded_to(aprev.id, ai)
                aprev = ai

        # Rebuild the topology angle and dihedral lists
        topology.rebuild()

class Homopolymer(Multiblock):
    """
    Species representing a linear homopolymer containing a chain of identical beads.
    """
    def __init__(self, id, mon, N, **kwargs):     
        super().__init__(id, [mon], [N], **kwargs)

class Diblock(Multiblock):
    """
    Species representing a diblock copolymer containing two blocks with different monomer types.
    """
    def __init__(self, id, amon, bmon, na, nb, **kwargs):     
        super().__init__(id, [amon, bmon], [na, nb], **kwargs)