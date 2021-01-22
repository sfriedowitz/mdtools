import numpy as np

#==============================================================================#

class MonomerType:
    """A class representing a chemical monomer type
    of a specific mass, charge, and size.

    Parameters
    ----------
    id : integer
        Unique ID for this monomer type.
    mass : float (keyword)
        Mass of this monomer.
    charge : float (keyword)
        Charge of this monomer.
    size : float (keyword)
        LJ diameter of this monomer (i.e. sigma).
    **kwargs : (optional)
        Extra keyword arguments to add to the monomer `attrib` dictionary.
    """
    def __init__(self, id, **kwargs):
        self.id     = id
        self.name   = kwargs.get("name", "")
        self.mass   = kwargs.get("mass", 1.0)
        self.charge = kwargs.get("charge", 0.0)
        self.size   = kwargs.get("size", 1.0)
        self.properties = {}

    def __repr__(self):
        return "MonomerType(id = {}, name = '{}')".format(self.id, self.name)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        else:
            return False

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return hash(self.id)

    def add_property(self, name, val):
        self.properties[name] = val

class Atom:
    """A specific instance of a monomer
    which can store position, velocity, and image data.

    Parameters
    ----------
    mon : MonomerType
        Monomer type this atom represents.
    mid : integer (keyword)
        ID of molecule this Atom is a part of.
    sid : integer (keyword)
        ID of species this Atom is a part of.
    """
    def __init__(self, mon, mid = -1, sid = -1, **kwargs):
        self.mon = mon
        self.id  = -1
        self.mid = mid
        self.sid = sid
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.img = np.zeros(3, dtype = "int")
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.id == other.id and self.mon == other.mon and 
                self.mid == other.mid and self.sid == other.sid)
        else:
            return False

    def __repr__(self):
        return "Atom(id = {}, mon = {}, mol = {}, species = {})".format(self.id, self.mon.id, self.mid, self.sid)
        
    def __hash__(self):
        return hash((self.id, self.mid, self.sid))
    
    def set_position(self, arr):
        assert(len(arr) == 3), "Invalid dimensions for Atom position vector."
        self.pos = np.array(arr)
        
    def set_velocity(self, arr):
        assert(len(arr) == 3), "Invalid dimensions for Atom velocity vector."
        self.vel = np.array(arr)

    def set_image(self, arr):
        assert(len(arr) == 3), "Invalid dimensions for Atom image vector."
        self.img = np.array(arr).astype("int")

#==============================================================================#
# Groups for bonds, angles, torsions
#==============================================================================#

class TopologyGroup:
    """A collection of atoms in a chemical functional unit.
    Parent class of Bonds, Angles, Dihedrals, and Impropers.

    Parameters
    ----------
    ids : integer (multiple)
        A variable number of integer IDs stored in the TopologyGroup.
    type : integer (keyword)
        Integer type ID for this group.
    """
    def __init__(self, *args, **kwargs):
        self.ids = tuple(args)
        self.type = kwargs.get("type", 1)
        
    def __getitem__(self, idx):
        return self.ids[idx]
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if not (len(self.ids) == len(other.ids)):
                return False
            else:
                for i in range(len(self.ids)):
                    if self.ids[i] == other.ids[i]:
                        continue
                    else:
                        return False  
                return True
        else:
            return False

    def __lt__(self, other):
        for i in range(len(self.ids)):
            a = self.ids[i]
            b = other[i]

            if a < b:
                return True
            elif a > b:
                return False

        return False
        
    def __hash__(self):
        return hash(self.ids)

    def size(self):
        return len(self.ids)
    
    def contains(self, id):
        return (id in self.ids)

#==============================================================================#

class Bond(TopologyGroup):
    def __init__(self, i, j, **kwargs):
        assert(i != j), "Repeated index in Bond."
        if i < j:
            super().__init__(i, j, **kwargs)
        else:
            super().__init__(j, i, **kwargs)

    def __repr__(self):
        return "Bond(type = {}, ids = [{}, {}])".format(self.type, self[0], self[1])

    def partner(self, id):
        assert self.contains(id), "Invalid partner id in Bond."
        if self.ids[0] == id:
            return self.ids[1]
        else:
            return self.ids[0]
        
class Angle(TopologyGroup):
    def __init__(self, i, j, k, **kwargs):
        assert(i != j and i !=k and j != k), "Repeated index in Angle."
        if i < k:
            super().__init__(i, j, k, **kwargs)
        else:
            super().__init__(k, j, i, **kwargs)

    def __repr__(self):
        return "Angle(type = {}, ids = [{}, {}, {}])".format(self.type, self[0], self[1], self[2])
        
class Dihedral(TopologyGroup):
    def __init__(self, i, j, k, l, **kwargs):
        assert(i != j and j != k and k != l), "Repeated index in Dihedral."
        if max(i, j) < max(k, l):
            super().__init__(i, j, k, l, **kwargs)
        else:
            super().__init__(l, k, j, i, **kwargs)

    def __repr__(self):
        return "Dihedral(type = {}, ids = [{}, {}, {}, {}])".format(self.type, self[0], self[1], self[2], self[3])

class Improper(TopologyGroup):
    def __init__(self, i, j, k, l, **kwargs):
        assert(i != j and j != k and k != l), "Repeated index in Improper."
        # i is taken to be the center atom
        # j/k/l are ordered increasing
        outer = sorted((j, k, l))
        super().__init__(i, outer[0], outer[1], outer[2], **kwargs)

    def __repr__(self):
        return "Improper(type = {}, ids = [{}, {}, {}, {}])".format(self.type, self[0], self[1], self[2], self[3])

    def __lt__(self, other):
        return True
