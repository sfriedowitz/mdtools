from .chemistry import *

#==============================================================================#

# TODO: How do we get the Topology to construct Angles/Dihdrals/Impropers with the proper type for LAMMPS data files?
#   We need to be able to specify this right now
#   For now, we can ignore since we will just use bonds for most things
#   Species must also carry information about Angle/Dihedral/Improper type field

# TODO: Topology is not structured well for removing/changing atom positions once created
#   Must changing indexing scheme to allow for deletion/insertion at will into the graph

class Topology:
    """A graph-like structure containing all Atoms,
    Bonds, Angles, Dihedrals, and Impropers in a system.

    Implements methods for adding atoms and bonds,
    and building higher order connections from this data
    (i.e. angles, dihedrals, impropers).
    """
    def __init__(self):
        self.__atoms = []
        self.__bonds = set()
        self.__angles = set()
        self.__dihedrals = set()
        self.__impropers = set()
        self.__bond_adj = []

        # Tallies of unique types for each component
        self.__bond_types = set()
        self.__angle_types = set()
        self.__dihedral_types = set()
        self.__improper_types = set()

    def __repr__(self):
        return "Topology({} atoms, {} bonds, {} angles, {} dihedrals, {} impropers)".format(
            self.natoms(), self.nbonds(), self.nangles(), self.ndihedrals(), self.nimpropers()
        )

    def __getitem__(self, idx):
        return self.atoms[idx] 
     
    # Containers
    @property
    def atoms(self):
        return self.__atoms
    
    @property
    def bonds(self):
        return self.__bonds
    
    @property
    def angles(self):
        return self.__angles
    
    @property
    def dihedrals(self):
        return self.__dihedrals
    
    @property
    def impropers(self):
        return self.__impropers
    
    @property
    def bond_adj(self):
        return self.__bond_adj

    @property
    def bond_types(self):
        return self.__bond_types

    @property
    def angle_types(self):
        return self.__angle_types

    @property
    def dihedral_types(self):
        return self.__dihedral_types

    @property
    def improper_types(self):
        return self.__improper_types
    
    def atom(self, id):
        return self.atoms[id-1]
    
    def atom_bonds(self, id):
        return self.bond_adj[id-1]

    # Property checks
    def natoms(self):
        return len(self.atoms)

    def nbonds(self):
        return len(self.bonds)

    def nangles(self):
        return len(self.angles)

    def ndihedrals(self):
        return len(self.dihedrals)

    def nimpropers(self):
        return len(self.impropers)

    def nbondtypes(self):
        return len(self.bond_types)

    def nangletypes(self):
        return len(self.angle_types)

    def ndihedraltypes(self):
        return len(self.dihedral_types)

    def nimpropertypes(self):
        return len(self.improper_types)

    def next_id(self):
        return self.natoms()+1
    
    def valid_id(self, id):
        return 1 <= id < self.next_id()
    
    def has_bond(self, id_a, id_b):
        assert self.valid_id(id_a) and self.valid_id(id_b)
        for b in self.atom_bonds(id_a):
            if b.contains(id_b):
                return True
        return False
    
    def clear(self):
        self.__atoms = []
        self.__bonds = set()
        self.__angles = set()
        self.__dihedrals = set()
        self.__impropers = set()
        self.__bond_adj = []
        self.__bond_types = set()
        self.__angle_types = set()
        self.__dihedral_types = set()
        self.__improper_types = set()
    
    # Adding and building
    def add_atom(self, a):
        a.id = self.next_id()
        self.atoms.append(a)
        self.bond_adj.append([])
        
    def add_bond(self, id_a, id_b, type = 1):
        assert id_a != id_b, "Cannot add bond between the same atoms."
        assert self.valid_id(id_a) and self.valid_id(id_b)
        
        if not self.has_bond(id_a, id_b):
            b = Bond(id_a, id_b, type = type)
            self.bonds.add(b)
            self.atom_bonds(id_a).append(b)
            self.atom_bonds(id_b).append(b)

        # Add the type to the set within topology
        self.bond_types.add(type)
    
    def add_atom_bonded_to(self, other_id, a, type = 1):
        a.id = self.next_id()
        self.atoms.append(a)
        self.bond_adj.append([])
        
        self.add_bond(a.id, other_id)

    def rebuild(self):
        self.rebuild_angles()
        self.rebuild_torsions()

    def rebuild_angles(self):
        for b1 in self.bonds:
            # Two atoms of this bond
            i = b1[0]
            j = b1[1]

            for b2 in self.atom_bonds(i): # Adjacency list for i, is the middle atom
                k = b2.partner(i)
                if k != j:
                    ang = Angle(k, i, j)
                    self.angles.add(ang)

            for b2 in self.atom_bonds(j): # Adjacency list for j, is the middle atom
                k = b2.partner(j)
                if k != i:
                    ang = Angle(i, j, k)
                    self.angles.add(ang)

    def rebuild_torsions(self):
        for ang in self.angles:
            # Three atoms of this angle
            i = ang[0]
            j = ang[1]
            k = ang[2]

            for b in self.atom_bonds(i): # Adjacency list for i
                l = b.partner(i)
                if l != j and l != k:
                    di = Dihedral(l, i, j, k)
                    self.dihedrals.add(di)

            for b in self.atom_bonds(k): # Adjacency list for k
                l = b.partner(k)
                if l != i and l != j:
                    di = Dihedral(i, j, k, l)
                    self.dihedrals.add(di)

            # This is an improper since j is the middle atom of the angle
            for b in self.atom_bonds(j): # Adjacency list for j
                l = b.partner(j)
                if l != i and l != k:
                    im = Improper(j, i, k, l)
                    self.impropers.add(im)