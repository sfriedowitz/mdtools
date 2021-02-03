import numpy as np
from MDAnalysis.lib._cutil import make_whole

# Module scripts
from .base import SingleGroupAnalysis, MultiGroupAnalysis
from .timeseries import DipoleTrajectory

# Helper functions
from ..lib.utils import zero_pad, bin_data, repair_molecules, save_path
from ..lib.correlation import correlation
from ..lib.fourier import fourier_integrate
from ..lib.distances import separation_array

#==============================================================================#

class DielectricBulk(SingleGroupAnalysis):
    """
    Compute the bulk dielectric constant at a given temperature 
    due to polarization fluctuations for atoms within a given group.

    Parameters
    ----------
    atomgroup : AtomGroup
        atom group to extract dipole fluctuations from
    temperature : real
        system temperature during simulation in Kelvin units (300.0)
    navg : int
        number of frames between updating averages of polarization and fluctuations

    Returns
    ----------
    results : dict
        dictionary containing extract dielectric constant and polarization components
    """
    def __init__(self, atomgroup, temperature = 300, navg = 1, repair = False, **kwargs):
        # Initialize parent class
        super().__init__(atomgroup, **kwargs)

        self.temperature = temperature
        self.repair = repair
        self.navg = np.maximum(1, int(navg))
        
    def _prepare(self):
        self.volume = 0.0
        self.P = np.zeros(3)
        self.P2 = np.zeros(3)
        self.charges = self._atomgroup.charges

        self.nframes = np.ceil((self.stop - self.start) / self.step).astype(int)
        self.time = np.round(self._trajectory.dt * np.arange(self.start, self.stop, self.step), decimals = 4)

        self.ncalc = 0
        self.eps_trj = np.zeros((self.nframes // self.navg, 5))

    def _single_frame(self):
        ag = self._atomgroup
        if self.repair:
            for frag in ag.fragments:
                make_whole(frag)

        self.volume += self._ts.volume

        P = np.dot(self.charges, self._atomgroup.positions)
        self.P += P
        self.P2 += P * P

        if self._frame_index % self.navg == 0:
            self._calculate_results()
            self.ncalc += 1

    def _calculate_results(self):
        index = self._frame_index + 1
        beta = 1. / (kB * self.temperature)
        pref = ec**2 / 1e-10

        self.results["P"] = self.P / index
        self.results["P2"] = self.P2 / index
        self.results["V"] = self.V / index

        self.results["fluct"] = self.results["P2"] - self.results["P"]**2
        self.results["eps"] = 1 + beta * pref * self.results["fluct"] / self.results["V"] / eps0
        self.results["eps_mean"] = self.results["eps"].mean()

        self.eps_trj[self.ncalc :] = np.hstack((
            self.time[self._frame_index], self.results["eps_mean"], self.results["eps"]
        ))

    def _conclude(self):
        self._calculate_results()
        self.results["eps_trj"] = self.eps_trj

        if self._verbose:
            print("The following averages for the complete trajectory have been calculated:\n")
            print("\t<P_x/y/z> = {:.4f}, {:.4f}, {:.4f} eÅ\n".format(
                self.results["P"][0], self.results["P"][1], self.results["P"][2]
            ))
            print("\t<P²_x/y/z> = {:.4f}, {:.4f}, {:.4f} (eÅ)²\n".format(
                self.results["P2"][0], self.results["P2"][1], self.results["P2"][2]
            ))

            print("\t<|P|²> = {:.4f} (eÅ)²\n".format( self.results["P2"].mean() ))
            print("\t|<P>|² = {:.4f} (eÅ)²\n".format( (self.results["P"]**2).mean() ))

            print("\t<|P|²> - |<P>|² = {:.4f} (eÅ)²\n".format( self.results["fluct"].mean() ))

            print("\tε_x/y/z = {:.2f}, {:.2f}, {:.2f}\n".format(
                self.results["eps"][0], self.results["eps"][1], self.results["eps"][2]
            ))

            print("\tε = {:.2f}".format(self.results["eps_mean"]))

    def save(self, prefix = "", delimiter = " ", **kwargs):
        output = save_path(prefix)
        if self._verbose:
            print("Saving results to files at location: {}".format(prefix))

        np.savetxt(output + "epsilon_bulk.txt", self.results["eps_trj"], 
            delimiter = delimiter, header = delimiter.join(["time", "eps", "eps_x", "eps_y", "eps_z"])
        )

#==============================================================================#
# Calculation of the spectrum for a polar-only system
#==============================================================================#

class DielectricSpectrumBulk(DipoleTrajectory):
    """
    Compute the linear dielectric response for a bulk system,
    assuming the polarization can be calculated without a special decomposition
    (i.e. applicable to a system of neutral, polar molecules like water). 

    Parameters
    ----------
    atomgroup : AtomGroup
        atom group to extract dipole fluctuations from
    temperature : real
        system temperature during simulation in Kelvin units (300.0)
    segs : int
        number of segments to average the polarization TCF over (1)
    nbins : int
        number of log-spaced bins to output spectrum in (200)
    binafter : int
        number of initial points before binning spectrum data (20)
    repair : bool
        boolean flag to repair molecules that cross periodic boundaries (False)
    
    Returns
    ----------
    results : dict
        dictionary containing extract susceptibility and polarization components
    """
    def __init__(self, atomgroup, temperature = 300, segs = 1, nbins = 200, binafter = 20, repair = False, **kwargs):
        # Initialize parent class
        super().__init__(atomgroup, labels = ["ag1"], restypes = ["NM"], repair = repair, **kwargs)

        self.temperature = temperature
        self.segs = segs

        self.nbins = nbins
        self.binafter = binafter

    def _prepare(self):
        # Add the basic stuff from parent class
        super()._prepare()

        # Specific traits for segmenting ACFs
        self.nframes_seg = self.nframes // self.segs
        self.results["time_seg"] = self.results["time"][:self.nframes_seg]

        # Check for binning
        self.bin = not (self.nframes_seg <= self.nbins)

    # TCF calculation
    def _calculate_corr(self):
        P = self.results["ag1_MD"]

        trj_len = self.nframes_seg * self.segs
        trjs = np.split(P[:trj_len], self.segs)

        corr = np.zeros(self.nframes_seg)
        for i in range(self.segs):
            corr += correlation(trjs[i], trjs[i])
        corr *= self.pref / self.segs

        self.results["corr"] = corr

    def _calculate_susc(self):
        time = self.results["time_seg"]
        corr = self.results["corr"]
        
        # Find when to stop integrating
        idx_neg = np.argwhere(corr < 0)
        if len(idx_neg) == 0:
            idx_int = corr.size
        else:
            idx_int = idx_neg[0, 0]

        # Numerically differentiate and FT directly
        corr_int = zero_pad(corr[:idx_int], corr.size)
        w, chi = fourier_integrate(time, corr_int)
        
        hn = len(w) // 2 + 1
        w = w[hn:]
        chi = chi[hn:]

        chi = corr[0] - 1j*w*chi

        self.results["omega"] = w
        self.results["susc"] = chi

    # Wrapping up
    def _conclude(self):
        self.results["volume"] = 1e-3 * self.volume / (self.nframes + 1)
        self.results["ag1_MD"] /= 10.0

        # Prefactor for autocorrelation functions
        self.pref = ec**2 * 1e9 / (3 * self.results["volume"] * kB * self.temperature * eps0)

        # Calculate the various time correlation functions needed
        if self._verbose:
            print("Calculating correlation functions and susceptibility...")

        self._calculate_corr()
        self._calculate_susc()

        # Bin all of the data to avoid having too many points
        if self.bin:
            if self._verbose:
                print("Binning data above datapoint {} in log-spaced bins.".format(self.binafter))
                print("Unbinned data consists of {} datapoints.".format(len(self.results["omega"])))

            self.results["omega"] = bin_data(self.results["omega"], self.nbins, after = self.binafter)
            self.results["susc"] = bin_data(self.results["susc"], self.nbins, after = self.binafter)
        else:
            if self._verbose:
                print("Not binning data of length {}.".format(len(self.results["omega"])))

    def save(self, prefix = "", delimiter = " ", **kwargs):
        output = save_path(prefix)
        if self._verbose:
            print("Saving results to files at location: {}".format(prefix))

        # Save the polarization ACF
        corr_arr = np.column_stack((self.results["time_seg"], self.results["corr"]))
        np.savetxt(output + "correlation.txt", corr_arr, delimiter = delimiter, 
            header = delimiter.join(["time", "corr"]))

        # Save the susceptibility
        susc_arr = np.column_stack((self.results["omega"], self.results["susc"].real, -1*self.results["susc"].imag))
        np.savetxt(output + "suscepibility.txt", susc_arr, delimiter = delimiter, 
            header =  delimiter.join(["omega", "susc_re", "susc_im"]))

#==============================================================================#
# Calculation for small ion decomposition
#==============================================================================#

class DielectricSpectrumIon(DipoleTrajectory):
    """
    Compute the dielectric spectrum for a system with mobile charges.
    Decomposes the spectrum into an solvent dipole contribution,
    a solvent-ion cross contribution, and a pure ion contribution.
    Implementation based on the analysis presented in
    'Rinne, K.F., Gekle, S., Netz, R.R. Journal of Chemical Physics. 149, 214502 (2014)'.

    Parameters
    ----------
    ag_sol : AtomGroup
        atom group of solvent molecules, assumed to be neutral solvent
    ag_ion : AtomGroup
        atom group of ions, assumed to be small ions
    temperature : real
        system temperature during simulation in Kelvin units (300.0)
    segs : int
        number of segments to average the polarization TCF over (1)
    nbins : int
        number of log-spaced bins to output spectrum in (200)
    binafter : int
        number of initial points before binning spectrum data (20)
    repair : bool
        boolean flag to repair molecules that cross periodic boundaries (False)
    
    Returns
    ----------
    results : dict
        dictionary containing extract susceptibility, conductivity, and polarization components
    """
    def __init__(self, ag_sol, ag_ion, temperature = 300, segs = 1, nbins = 200, binafter = 20, repair = False, **kwargs):
        # Initialize parent class
        super().__init__(ag_sol, ag_ion, labels = ["sol", "ion"], restypes = ["NM", "SP"], repair = repair, **kwargs)

        self.temperature = temperature
        self.repair = repair
        self.segs = segs

        self.nbins = nbins
        self.binafter = binafter

    def _prepare(self):
        # Add the basic stuff from parent class
        super()._prepare()

        # Specific traits for segmenting ACFs
        self.nframes_seg = self.nframes // self.segs
        self.results["time_seg"] = self.results["time"][:self.nframes_seg]

        # Check for binning
        self.bin = not (self.nframes_seg <= self.nbins)

    # Computes time correlation functions
    def _calculate_corr_water(self):
        M = self.results["sol_MD"]

        trj_len = self.nframes_seg * self.segs
        trjs = np.split(M[:trj_len], self.segs)

        corr = np.zeros(self.nframes_seg)
        for i in range(self.segs):
            corr += correlation(trjs[i], trjs[i])
        corr *= self.pref / self.segs

        self.results["corr_w"] = corr

    def _calculate_corr_ion_water(self):
        M = self.results["sol_MD"]
        J = self.results["ion_J"]

        trj_len = self.nframes_seg * self.segs
        trjs_M = np.split(M[:trj_len], self.segs)
        trjs_J = np.split(J[:trj_len], self.segs)

        corr = np.zeros(self.nframes_seg)
        for i in range(self.segs):
            corr += correlation(trjs_M[i], trjs_J[i]) - correlation(trjs_J[i], trjs_M[i])
        corr *= 0.5 * self.pref / self.segs

        self.results["corr_iw"] = corr

    def _calculate_corr_ion(self):
        J = self.results["ion_J"]

        trj_len = self.nframes_seg * self.segs
        trjs = np.split(J[:trj_len], self.segs)

        corr = np.zeros(self.nframes_seg)
        for i in range(self.segs):
            corr += correlation(trjs[i], trjs[i])
        corr *= self.pref / self.segs

        self.results["corr_i"] = corr

    # Transform to susceptibility
    def _calculate_susc_water(self):
        time = self.results["time_seg"]
        corr = self.results["corr_w"]
        
        # Find when to stop integrating
        idx_neg = np.argwhere(corr < 0)
        if len(idx_neg) == 0:
            idx_int = corr.size
        else:
            idx_int = idx_neg[0, 0]

        # Numerically differentiate and FT directly
        corr_int = zero_pad(corr[:idx_int], corr.size)
        w, chi = fourier_integrate(time, corr_int)
        
        hn = len(w) // 2 + 1
        w = w[hn:]
        chi = chi[hn:]
        chi = corr[0] - 1j*w*chi

        self.results["omega"] = w
        self.results["susc_w"] = chi

    def _calculate_susc_ion(self):
        time = self.results["time_seg"]
        corr_iw = self.results["corr_iw"]
        corr_i = self.results["corr_i"]
        
        # Find integration cutoffs
        # Ion-water correlation
        idx_max = np.argmax(corr_iw[:500]) # Find peak within first 500 points
        idx_negs = np.argwhere(corr_iw[idx_max:] < 0) # Find negative crossing after the peak
        if len(idx_negs) == 0:
           idx_int = corr_iw.size
        else:
           idx_int = np.min(idx_negs) + idx_max
        corr_iw_int = zero_pad(corr_iw[:idx_int], length = corr_iw.size)

        # Ion-ion correlation
        t_int = 3.0 # Hard coded @ 3 ps for now
        idx_int = np.argmin(np.abs(time - t_int))
        corr_i_int = zero_pad(corr_i[:idx_int], length = corr_i.size)

        # Calculate the susceptibility contributions
        w, chi_iw = fourier_integrate(time, corr_iw_int)
        chi_i = fourier_integrate(time, corr_i_int, indvar = False)

        # Determine the conductivity
        hn = len(w) // 2 + 1
        w = w[hn:]
        sig_i = chi_i[hn:]
        sig_iw = -1j * k * chi_iw[hn:] 
        sig = sig_i + sig_iw

        # And scale the susceptibility
        chi_iw = -2 * chi_iw[hn:]
        chi_i = -(1j / w) * (chi_i[hn:] - sig[0].real)
        
        self.results["susc_iw"] = chi_iw
        self.results["susc_i"] = chi_i
        self.results["cond"] = sig

    # Wrapping up
    def _conclude(self):
        self.results["volume"] = 1e-3 * self.volume / (self.nframes + 1)
        self.results["sol_MD"] /= 10.0
        self.results["ion_J"]  /= 10.0

        # Prefactor for autocorrelation functions
        self.pref = ec**2 * 1e9 / (3 * self.results["volume"] * kB * self.temperature * eps0)

        # Calculate the various time correlation functions needed
        if self._verbose:
            print("Calculating time correlation functions...")

        self._calculate_corr_water()
        self._calculate_corr_ion_water()
        self._calculate_corr_ion()

        # Calculate the susceptibilities from ACFs
        if self._verbose:
            print("Calculating susceptibilities...")

        self._calculate_susc_water()
        self._calculate_susc_ion()

        self.results["susc"] = self.results["susc_w"] + self.results["susc_iw"] + self.results["susc_i"]

        # Bin all of the data to avoid having too many points
        if self.bin:
            if self._verbose:
                print("Binning data above datapoint {} in log-spaced bins.".format(self.binafter))
                print("Unbinned data consists of {} datapoints.".format(len(self.results["omega"])))
            self.results["omega"]   = bin_data(self.results["omega"], self.nbins, after = self.binafter)
            self.results["susc_w"]  = bin_data(self.results["susc_w"], self.nbins, after = self.binafter)
            self.results["susc_iw"] = bin_data(self.results["susc_iw"], self.nbins, after = self.binafter)
            self.results["susc_i"]  = bin_data(self.results["susc_i"], self.nbins, after = self.binafter)
            self.results["susc"]    = bin_data(self.results["susc"], self.nbins, after = self.binafter)
            self.results["cond"]    = bin_data(self.results["cond"], self.nbins, after = self.binafter)
        else:
            if self._verbose:
                print("Not binning data of length {}.".format(len(self.results["omega"])))

    def save(self, prefix = "", delimiter = " ", **kwargs):
        # Saves the ion trajectories and stuff
        super().save(prefix = prefix, delimiter = delimiter, **kwargs)
        output = save_path(prefix)

        # Save each of the ACFs
        corr_arr = np.column_stack((self.results["time_seg"], self.results["corr_w"], self.results["corr_iw"], self.results["corr_i"]))
        np.savetxt(output + "correlation.dat", corr_arr, delimiter = delimiter, 
            header = delimiter.join(["time", "corr_w", "corr_iw", "corr_i"])

        # Save the susceptibilities and conductivity
        header = ["omega", "susc_re", "susc_im", "susc_w_re", "susc_w_im", "susc_iw_re", "susc_iw_im", "susc_i_re", "susc_i_im"]
        susc_arr = np.column_stack((
                self.results["omega"],
                self.results["susc"].real,    -1*self.results["susc"].imag,
                self.results["susc_w"].real,  -1*self.results["susc_w"].imag,
                self.results["susc_iw"].real, -1*self.results["susc_iw"].imag,
                self.results["susc_i"].real,  -1*self.results["susc_i"].imag
            ))
        np.savetxt(output + "susceptibility.dat", susc_arr, delimiter = delimiter, header = delimeter.join(header))

        cond_arr = np.column_stack((self.results["omega"], self.results["cond"].real, -1*self.results["cond"].imag))
        np.savetxt(output + "conductivity.dat", cond_arr, delimiter = delimiter, 
            header = delimiter.join(["omega", "cond_re", "cond_im"]))