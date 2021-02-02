import numpy as np

# System builder tool
from ..system.chemistry import *
from ..system.species import *
from ..system.system import *

#========================================================================================#

def build_simple_electrolyte(ncat, nan, zcat, zan, L, fill_solvent = 0.0, fname = "init.data"):
    """
    Build a LAMMPS data file for a molecular system containing oppositely charged small ions
    and optional neutral solvent.
    """
    if zan > 0.0 or zcat < 0.0:
        raise ValueError("Monomer charges on +/- type are incorrect sign.")

    # Create the monomers
    catmon = MonomerType(1, size = 1.0, charge = zcat)
    anmon = MonomerType(2, size = 1.0, charge = zan)

    # Create the species
    cation = Point(1, catmon)
    anion = Point(2, anmon)

    # Create the box, system, and add species
    box = Box([L, L, L])
    sys = MolecularSystem(box)

    # Add the species
    sys.add_species(cation, nmol = ncat)
    sys.add_species(anion, nmol = nan)

    if fill_solvent > 0.0:
        solvmon = MonomerType(3, size = 1.0, charge = 0.0)
        solvent = Point(3, solvmon)
        sys.add_species(solvent, fill = fill_solvent)

    sys.generate_molecules()

    # Write the output data
    sys.write_lammps_data(fname = fname, bonds = True)

def build_single_polyion(npol, nsalt, dp, zpol, L, fill_solvent = 0.0, bscale = 1.25, fname = "init.data"):
    """
    Build a LAMMPS data file for a molecular system containing a single polyelectrolyte species,
    counterions, and optional added salt neutral solvent.
    """

    # Create the monomers
    polmon = MonomerType(1, size = 1.0, charge = zpol)
    catmon = MonomerType(2, size = 1.0, charge = 1.0)
    anmon = MonomerType(3, size = 1.0, charge = -1.0)

    # Create the species
    polyion = Homopolymer(1, polmon, dp, bond_scale = bscale)
    cation = Point(2, catmon)
    anion = Point(3, anmon)

    # Determine numbers of ions based on polymer charge
    pol_charge = npol * polyion.charge()
    if pol_charge != np.rint(pol_charge):
        raise ValueError("Polyion contains a fractional total charge.")

    ncat = nsalt
    nan = nsalt
    if pol_charge < 0.0:
        ncat += np.rint(np.abs(pol_charge))
    elif pol_charge > 0.0:
        nan += np.rint(np.abs(pol_charge))

    # Create the box, system, and add species
    box = Box([L, L, L])
    sys = MolecularSystem(box)

    species = [polyion, cation, anion]
    counts = [npol, ncat, nan]
    for i, sp in enumerate(species):
        nmol = counts[i]
        sys.add_species(sp, nmol = nmol)

    if fill_solvent > 0.0:
        solvmon = MonomerType(4, size = 1.0, charge = 0.0)
        solvent = Point(4, solvmon)
        sys.add_species(solvent, fill = fill_solvent)

    sys.generate_molecules()

    # Write the output data
    sys.write_lammps_data(fname = fname, bonds = True)


def build_multi_polyion(na, nc, ns, dpa, dpc, za, zc, L, counter = False, fill_solvent = 0.0, 
    center_scale = 1.0, bond_scale = 1.25, fname = "init.data"):
    """
    Build a LAMMPS data file for a molecular system containing oppositely charged polyelectrolytes,
    optional counterions, optional added salt, and optional neutral solvent.
    """

    # Quick check
    if za > 0.0 or zc < 0.0:
        raise ValueError("Monomer charges on A/C type are incorrect sign.")

    # Create the monomers
    amon = MonomerType(1, size = 1.0, charge = za)
    cmon = MonomerType(2, size = 1.0, charge = zc)
    pmon = MonomerType(3, size = 1.0, charge = 1.0)
    mmon = MonomerType(4, size = 1.0, charge = -1.0)

    # Create the species
    def rinit(box, scale):
        center = box.dimensions[:3]/2.0
        return center + scale*center*(2*np.random.rand(3) - 1)
    rfunc = lambda b: rinit(b, center_scale)

    pa = Homopolymer(1, amon, dpa, bond_scale = bond_scale, initializer = rfunc)
    pc = Homopolymer(2, cmon, dpc, bond_scale = bond_scale, initializer = rfunc)
    cat = Point(3, pmon)
    an = Point(4, mmon)

    species = [pa, pc, cat, an]
    counts = [na, nc, ns, ns]

    # Figure out the necessary charge balancing, add separate counterion species if necessary
    pa_charge = np.abs(na * pa.charge())
    pc_charge = np.abs(nc * pc.charge())

    if pa_charge != np.rint(pa_charge):
        raise ValueError("Polyanion contains a fractional charge.")
    if pc_charge != np.rint(pc_charge):
        raise ValueError("Polycation contains a fractional charge.")

    if counter:
        counter_cat = Point(5, pmon, initializer = rfunc)
        counter_an = Point(6, mmon, initializer = rfunc)

        species.extend([counter_cat, counter_an])
        counts.extend([pa_charge, pc_charge])
    else:
        ngap = np.abs(pa_charge - pc_charge)

        counter_mon = pmon if pa_charge > pc_charge else mmon
        counterion = Point(5, counter_mon, initializer = rfunc)

        species.append(counterion)
        counts.append(ngap)

    # Create the box, system, and add species
    box = Box([L, L, L])
    sys = MolecularSystem(box)

    species = [pa, pc, cat, an]
    counts = [na, nc, ncat, nan]
    for i, sp in enumerate(species):
        nmol = counts[i]
        sys.add_species(sp, nmol = nmol)

    if fill_solvent > 0:
        solvmon = MonomerType(5, size = 1.0, charge = 0.0)
        solvent = Point(5, solvmon)
        sys.add_species(solvent, fill = fill_solvent)

    sys.generate_molecules()

    # Write the output data
    sys.write_lammps_data(fname = fname, bonds = True)

def build_coacervate_layer(na, nc, ns, dpa, dpc, za, zc, lxy, lz, zscale, counter = False, bond_scale = 1.25, fname = "init.data"):
    """
    Build a LAMMPS data file for a molecular system containing oppositely charged polyelectrolytes,
    optional counterions, optional added salt, and optional neutral solvent.
    The polymer positions are initialized in a layer based on the X-Y dimensions of the box,
    while the Z dimension is assumed to be extended (i.e. tetragonal box).
    """

    # Quick check
    if za > 0.0 or zc < 0.0:
        raise ValueError("Monomer charges on A/C type are incorrect sign.")

    # Create the monomers
    amon = MonomerType(1, size = 1.0, charge = za)
    cmon = MonomerType(2, size = 1.0, charge = zc)
    pmon = MonomerType(3, size = 1.0, charge = 1.0)
    mmon = MonomerType(4, size = 1.0, charge = -1.0)

    # Create the species
    # Places randomly in a box centered at l0 with side lengths of l0/2
    def rinit(box, zscale):
        center = 0.5*box.dimensions[:3]
        scale = np.array([0.9, 0.9, 1.0/zscale])
        return center + scale*center*(2*np.random.rand(3) - 1)
    rfunc = lambda b: rinit(b, zscale)

    pa = Homopolymer(1, amon, dpa, bond_scale = bond_scale, initializer = rfunc)
    pc = Homopolymer(2, cmon, dpc, bond_scale = bond_scale, initializer = rfunc)
    cat = Point(3, pmon)
    an = Point(4, mmon)

    species = [pa, pc, cat, an]
    counts = [na, nc, ns, ns]

    # Figure out the necessary charge balancing, add separate counterion species if necessary
    pa_charge = np.abs(na * pa.charge())
    pc_charge = np.abs(nc * pc.charge())

    if pa_charge != np.rint(pa_charge):
        raise ValueError("Polyanion contains a fractional charge.")
    if pc_charge != np.rint(pc_charge):
        raise ValueError("Polycation contains a fractional charge.")

    if counter:
        counter_cat = Point(5, pmon, initializer = rfunc)
        counter_an = Point(6, mmon, initializer = rfunc)

        species.extend([counter_cat, counter_an])
        counts.extend([pa_charge, pc_charge])
    else:
        ngap = np.abs(pa_charge - pc_charge)

        counter_mon = pmon if pa_charge > pc_charge else mmon
        counterion = Point(5, counter_mon, initializer = rfunc)

        species.append(counterion)
        counts.append(ngap)

    # Create the box and system
    box = Box([lxy, lxy, lz])
    sys = MolecularSystem(box)

    for i, sp in enumerate(species):
        nmol = counts[i]
        sys.add_species(sp, nmol = nmol)

    sys.generate_molecules()

    # Write the output data
    sys.write_lammps_data(fname = fname, bonds = True)

def build_diblock(npol, na, nb, za, zb, L, fill_solvent = 0.0, bscale = 1.25, fname = "init.data"):
    """
    Build a LAMMPS data file for a molecular system containing diblock copolymers
    (with optional monomer charges) and optional neutral solvent.
    """

    # Create the monomers
    amon = MonomerType(1, size = 1.0, charge = za)
    bmon = MonomerType(2, size = 1.0, charge = zb)

    diblock = Diblock(1, amon, bmon, na, nb, bond_scale = bscale)

    box = Box([L, L, L])
    sys = MolecularSystem(box)
    sys.add_species(diblock, nmol = npol)

    if fill_solvent > 0:
        solvmon = MonomerType(3, size = 1.0, charge = 0.0)
        solvent = Point(2, solvmon)
        sys.add_species(solvent, fill = fill_solvent)

    sys.generate_molecules()

    # Write the output data
    sys.write_lammps_data(fname = fname, bonds = True)