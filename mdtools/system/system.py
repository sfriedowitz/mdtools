from __future__ import division, print_function
import numpy as np
import datetime
import warnings

# Drawing tools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Custom classes
from .box import *
from .chemistry import *
from .topology import *
from .species import *

class MolecularSystem:
    """A holder class containing all information for building a molecular system.
    Contains a simulation box and topology.

    Implements methods for generating atoms, visualizing the configuration, 
    and writing data in a standard output format.

    Parameters
    ----------
    box : Box
        Simulation box for the system
    """
    def __init__(self, box):
        self.box = box
        self.topology = Topology()

        self.monomers = set()
        self.species = {}
        self.generate_nmol = {}
        self.generated = False

    def __repr__(self):
        return "MolecularSystem(V = {:.2f}, {} species)".format(self.volume(), self.nspecies())

    def volume(self):
        return self.box.volume()

    def nspecies(self):
        return len(self.species)

    def nmonomers(self):
        return len(self.monomers)

    def natoms(self):
        return self.topology.natoms()

    def add_species(self, species, **kwargs):
        if species.id in self.species:
            warnings.warn("Replacing species with id = {} in system.".format(species.id), RuntimeWarning)

        if not "nmol" in kwargs and not "fill" in kwargs:
            raise ValueError("One of either `nmol` or `fill` keyword arguments must be specified.")
        elif "nmol" in kwargs and "fill" in kwargs:
            raise ValueError("Only one of either `nmol` or `fill` keyword arguments should be specified.")

        if "nmol" in kwargs:
            self.generate_nmol[species.id] = int(kwargs.get("nmol"))

        elif "fill" in kwargs:
            fill = kwargs.get("fill")
            assert 0.0 < fill < 1.0, "Fill fraction must be between zero and unity."

            # Target volume of all monomers
            vol_target = fill * self.box.volume()

            # Get the current amount of volume taken up by other species
            vol_curr = 0.0
            for (sid, other) in self.species.items():
                vol_curr += self.generate_nmol[sid] * other.volume()

            vol_left = vol_target - vol_curr # Amount needed to fill
            nmol_fill = np.rint(vol_left / species.volume())
            nmol_fill = np.maximum(0, nmol_fill)

            self.generate_nmol[species.id] = int(nmol_fill)

        # Add the species to the system after the other stuff has worked out already
        self.species[species.id] = species
        for mon in species.monomers:
            self.monomers.add(mon)

    # Molecule generation and placement in the topology
    def _expected_charge(self):
        charge_gen = 0.0
        for (sid, species) in self.species.items():
            nmol = self.generate_nmol[sid]
            charge_gen += nmol * species.charge()
        return charge_gen

    def _expected_volume(self):
        vol_gen = 0.0
        for (sid, species) in self.species.items():
            nmol = self.generate_nmol[sid]
            vol_gen += nmol * species.volume()
        return vol_gen
    
    def generate_molecules(self, check_volume = True, check_charge = True):
        # Safety checks on charge and volume
        vol_gen = self._expected_volume()
        charge_gen = self._expected_charge()
        if check_volume and vol_gen > self.box.volume():
            raise ValueError("Volume of molecules exceeds box volume.")
        if check_charge and not np.isclose(charge_gen, 0.0):
            raise ValueError("Net charge of molecules not equal to zero.")

        # Reset the internal topology
        self.topology.clear()

        # Actually generate the molecules
        # Add safety checks later on for space filling and overlaps????
        mid0 = 0
        for (sid, species) in self.species.items():
            nmol = self.generate_nmol[sid]
            if nmol > 0:
                species.generate(nmol, mid0, self.topology, self.box)
                mid0 += nmol

    # Plotting and visualization of system
    def _draw_box(self, ax):        
        points = np.array([
            [0.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0], 
            [1.0, 1.0, 0.0], 
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], 
            [1.0, 0.0, 1.0], 
            [1.0, 1.0, 1.0], 
            [0.0, 1.0, 1.0]
        ])
        Z = np.zeros((8,3))
        for i in range(8): Z[i,:] = np.dot(points[i,:],self.box.h)

        # Scaled polygon sides
        verts = [[Z[0],Z[1],Z[2],Z[3]],
         [Z[4],Z[5],Z[6],Z[7]], 
         [Z[0],Z[1],Z[5],Z[4]], 
         [Z[2],Z[3],Z[7],Z[6]], 
         [Z[1],Z[2],Z[6],Z[5]],
         [Z[4],Z[7],Z[3],Z[0]]]

        ax.add_collection3d(Poly3DCollection(verts, linewidths=1, edgecolors='k', alpha=0.01))

    def _draw_molecules(self, ax):
        # TODO: This is quite slow right now
        # Can we speed up the individual drawing of all bonds and atoms
        #   by a single call to scatter3D/plot??

        nmon = self.nmonomers()
        cm = plt.get_cmap('gist_rainbow')
        colors = [cm(1.*i/nmon) for i in range(nmon)]
        color_dict = {mon.id : colors[i] for i, mon in enumerate(self.monomers)}

        # Plot all the individual atoms
        for ai in self.topology.atoms:
            mon = ai.mon
            # Unwrap the atom position
            pos = ai.pos + np.dot(self.box.h, ai.img)
            ax.scatter3D(pos[0], pos[1], pos[2], c = np.array([color_dict[mon.id]]), 
                s = 200 * mon.size, edgecolors = 'black')

        # Plot all the bonds between atoms
        for bi in self.topology.bonds:
            a1 = self.topology.atom(bi[0])
            a2 = self.topology.atom(bi[1])

            # Unwrap the positions
            pos1 = a1.pos + np.dot(self.box.h, a1.img)
            pos2 = a2.pos + np.dot(self.box.h, a2.img)
            points = np.array([pos1, pos2])

            # Plot line segment
            ax.plot(points[:,0], points[:,1], points[:,2], c = color_dict[a1.mon.id])

    def draw(self, figsize = (10, 8), **kwargs):
        elev = kwargs.get("elev", 30)
        azim = kwargs.get("azim", -60)

        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111, projection='3d', elev = elev, azim = azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Draw box and molecules
        self._draw_box(ax)
        self._draw_molecules(ax)

    # File IO methods
    def write_lammps_data(self, fname = None, **kwargs):
        """
        Write the system structure and topology as a LAMMPS data file.
        """

        # Keyword flags
        write_bonds = kwargs.get("bonds", False)
        write_angles = kwargs.get("angles", False)
        write_dihedrals = kwargs.get("dihedrals", False)
        write_impropers = kwargs.get("impropers", False)

        # Meta info on system
        box = self.box
        top = self.topology

        nmons = self.nmonomers()
        natoms = self.natoms()
        nbonds = top.nbonds()
        nbond_types = top.nbondtypes()

        # Create script header and system info
        now = datetime.datetime.now()
        header = "# LAMMPS data file created on {}\n\n".format(str(now.strftime("%Y-%m-%d %H:%M")))

        sys_info = ("{} atoms" "\n"
                    "{} atom types" "\n"
                    "{} bonds" "\n"
                    "{} bond types" "\n\n"
                    ).format(natoms, nmons, nbonds, nbond_types)

        # Write the box dimensions header
        if box.is_orthogonal():
            box_info = ("{0:<15.12f} {1:<15.12f} xlo xhi" "\n"
                        "{0:<15.12f} {2:<15.12f} ylo yhi" "\n"
                        "{0:<15.12f} {3:<15.12f} zlo zhi" "\n\n"
                        ).format(0, *box.dimensions[:3])
        else:
            # Must add the triclinic tilt factors to the file
            h = box.h
            box_info = ("{0:<15.12f} {1:<15.12f} xlo xhi" "\n"
                        "{0:<15.12f} {2:<15.12f} ylo yhi" "\n"
                        "{0:<15.12f} {3:<15.12f} zlo zhi" "\n"
                        "{4:<15.12f} {5:<15.12f} {6:<15.12f} xy xz yz" "\n\n"
                        ).format(0, h[0,0], h[1,1], h[2, 2], h[0,1], h[0,2], h[1,2])

        
        mass_info = "Masses\n\n"
        for mon in self.monomers:
            mass_info += "{} {}\n".format(mon.id, mon.mass)
        
        # Atom section
        atom_info = "\nAtoms\n\n"
        for i, atom in enumerate(top.atoms):
            mon = atom.mon
            pos = atom.pos
            img = atom.img
            atom_info += "{:.0f} {:.0f} {:.0f} {:.13f} {:.13e} {:.13e} {:.13e} {:.0f} {:.0f} {:.0f}\n".format(
                atom.id, atom.mid, mon.id, mon.charge,
                pos[0], pos[1], pos[2], img[0], img[1], img[2]
            )

        # Bond section
        if write_bonds and top.nbonds() > 0:
            bond_info = "\nBonds\n"
            for i, bond in enumerate(top.bonds):
                bond_info += "\n{} {} {} {}".format(i+1, bond.type, bond[0], bond[1])
        else:
            bond_info = ""

        # Angle section
        if write_angles and top.nangles() > 0:
            angle_info = "\nAngles\n"
            for i, ang in enumerate(top.angles):
                angle_info += "\n{} {} {} {} {}".format(i+1, ang.type, ang[0], ang[1], ang[2])
        else:
            angle_info = ""

        # Dihedral section
        if write_dihedrals and top.ndihedrals() > 0:
            dihedral_info = "\nDihedrals\n"
            for i, dih in enumerate(top.dihedrals):
                dihedral_info += "\n{} {} {} {} {} {}".format(i+1, dih.type, dih[0], dih[1], dih[2], dih[3])
        else:
            dihedral_info = ""

        # Improper section
        if write_impropers and top.nimpropers() > 0:
            improper_info = "\nImpropers\n"
            for i, imp in enumerate(top.impropers):
                improper_info += "\n{} {} {} {} {} {}".format(i+1, imp.type, imp[0], imp[1], imp[2], imp[3])
        else:
            improper_info = ""
        
        # Piece together the final script
        script = header + sys_info + box_info + mass_info + atom_info
        script += bond_info + angle_info + dihedral_info + improper_info
        
        if fname is not None:
            with open(fname, "w") as file: file.write(script)

        return script
