import scipy.constants

#==============================================================================#

ec = scipy.constants.elementary_charge
kB = scipy.constants.Boltzmann
eps0 = scipy.constants.epsilon_0
Navo = scipy.constants.Avogadro

#==============================================================================#

def molar2rho(val):
    return val * 1000.0 * Navo * 1e-27

def rho2molar(val):
    return val / 1000.0 / Navo / 1e-27

def molar2box(conc, nmon):
    return (nmon / molar2rho(conc))**(1/3)

def rho2box(rho, nmon):
    return (nmon / rho)**(1/3)

def box2rho(box, nmon):
    return nmon / box**3