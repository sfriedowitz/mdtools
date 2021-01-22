import numpy as np

# Module scripts
from .base import SingleGroupAnalysis, MultiGroupAnalysis
from .timeseries import DipoleTrajectory

# Helper functions
from ..lib.utils import zero_pad, bin_data, repair_molecules, save_path
from ..lib.correlation import correlations
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
        self.V = 0.0
        self.P = np.zeros(3)
        self.P2 = np.zeros(3)
        self.charges = self._atomgroup.charges

        self.nframes = np.ceil((self.stop - self.start) / self.step).astype(int)
        self.time = np.round(self._trajectory.dt * np.arange(self.start, self.stop, self.step), decimals = 2)

        self.ncalc = 0
        self.eps_trj = np.zeros((self.nframes // self.outfreq, 5))

    def _single_frame(self):
        ag = self._atomgroup
        if self.repair:
            repair_molecules(ag)

        self.V += self._ts.volume

        P = np.dot(self.charges, self._atomgroup.positions)
        self.P += P
        self.P2 += P * P

        if self._frame_index % self.outfreq == 0:
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
        self.output = save_path(prefix)
        if self._verbose:
            print("Saving results to files at location: {}".format(prefix))

        np.savetxt(self.output + "epsilon_bulk.txt", self.results["eps_trj"], 
            delimiter = delimiter, header = delimiter.join(["time", "eps", "eps_x", "eps_y", "eps_z"])
        )

#==============================================================================#
# Calculation of the spectrum for a polar-only system
#==============================================================================#

class DielectricSpectrumBulk(DipoleTrajectory):
    """
    Compute the linear dielectric response for a bulk system,
    assuming the polarization can be calculated without a special decompositions
    (i.e. applicable to polar water/solvent). 
    Assumes the solvent is a neutral molecule for calculating the polarization trajectory.

    :param temperature (float): Temperature (K)
    :param repair (bool): Make broken molecules whole again
        (only works if molecule is smaller than shortest box vector)
    :param segs (int): Number of trajectory segments to average the ACFs over
    :param bins (int): Determines the number of bins used for data averaging;
                           (this parameter sets the upper limit).
                           The data are by default binned logarithmically.
                           This helps to reduce noise, particularly in
                           the high-frequency domain, and also prevents plot
                           files from being too large.
    :param binafter (int): The number of low-frequency data points that are left unbinned.
    :param output (str): Prefix for output filenames

    :returns (dict): Results dictionary containing
        * time : Time intervals of the trajectory, up to ACF length
        * freq : Frequency of the Fourier space components
        * volume : Average system volume across the trajectory
        * acf  : Polarization ACF
        * susc : Polarization contribution to susceptibilty
    """
    def __init__(self, atomgroup, restype = None, temperature = 300, repair = False, 
        segs = 1, bins = 200, binafter = 20, **kwargs):
        # Initialize parent class
        super().__init__(atomgroup, labels = ["SOL"], restypes = ["NM"], repair = repair, **kwargs)

        self.temperature = temperature
        self.repair = repair
        self.segs = segs

        self.bins = bins
        self.binafter = binafter

    def _prepare(self):
        # Add the basic stuff from parent class
        super()._prepare()

        # Specific traits for segmenting ACFs
        self.nframes_seg = self.nframes // self.segs
        self.results["time_seg"] = self.results["time"][:self.nframes_seg]

        # Check for binning
        self.bin = not (self.nframes_seg <= self.bins)

    #==============================================================================# 
    # ACF and susceptibility calculations
    #==============================================================================# 

    def _calculate_acf(self):
        P = self.results["SOL_MD"]

        trj_len = self.nframes_seg * self.segs
        trjs = np.split(P[:trj_len], self.segs)

        acf = np.zeros(self.nframes_seg)
        for i in range(self.segs):
            acf += vcorrelate(trjs[i], trjs[i])
        acf *= self.pref / self.segs

        self.results["acf"] = acf

    def _calculate_susc(self):
        corr = self.results["acf"]
        time = self.results["time_seg"]
        
        # Find when to stop integrating
        idx_neg = np.argwhere(corr < 0)
        if len(idx_neg) == 0:
            idx_int = corr.size
        else:
            idx_int = idx_neg[0, 0]

        # Numerically differentiate and FT directly
        corr_int = zero_pad(corr[:idx_int], length = corr.size)
        corr_deriv = - np.append(np.diff(corr_int) / self.results["dt"], [0.0])
        
        # Do the Laplace FFT
        chi = FT(time, corr_deriv, indvar = False)
        
        # Piece together what we need
        halfN = chi.size // 2 + 1
        chi = chi[halfN:]

        self.results["susc"] = chi

    #==============================================================================#
    # Wrapping up
    #==============================================================================#

    def _conclude(self):
        self.results["freq"] = np.fft.rfftfreq(
            self.results["time_seg"].size, d = self.results["dt"]
        )[1:-1]

        self.results["volume"] = 1e-3 * self.volume / (self.nframes + 1)
        self.results["SOL_MD"] /= 10.0

        # Prefactor for autocorrelation functions
        self.pref = ec**2 * 1e9 / (3 * self.results["volume"] * kB * self.temperature * eps0)

        # Calculate the various time correlation functions needed
        if self._verbose:
            print("Calculating correlation functions and susceptibility...")
        self._calculate_acf()
        self._calculate_susc()

        # Bin all of the data to avoid having too many points
        if self.bin:
            if self._verbose:
                print("Binning data above datapoint {} in log-spaced bins.".format(self.binafter))
                print("Unbinned data consists of {} datapoints.".format(len(self.results["freq"])))

            self.results["freq"] = bin_data(self.results["freq"], self.bins, after = self.binafter)
            self.results["susc"] = bin_data(self.results["susc"], self.bins, after = self.binafter)
        else:
            if self._verbose:
                print("Not binning data of length {}.".format(len(self.results["freq"])))

    def save(self, prefix = "", delimiter = " ", **kwargs):
        prefix = prefix + "_" if prefix else ""
        if self._verbose:
            print("Saving results to files at location: {}".format(prefix))

        # Save the polarization ACF
        np.savetxt(prefix + "acf.txt", self.results["acf"], delimiter = delimiter, 
            header = delimiter.join(["time_seg", "acf"]))

        # Save the susceptibility
        np.savetxt(prefix + "susceptibility.txt",
            np.transpose((self.results["freq"], self.results["susc"].real, -1*self.results["susc"].imag)),
            delimiter = delimiter, header =  delimiter.join(["freq", "susc_re", "susc_im"])
        )

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

    :param temperature (float): Temperature (K)
    :param repair (bool): Make broken molecules whole again
        (only works if molecule is smaller than shortest box vector)
    :param segs (int): Number of trajectory segments to average the ACFs over
    :param bins (int): Determines the number of bins used for data averaging;
                           (this parameter sets the upper limit).
                           The data are by default binned logarithmically.
                           This helps to reduce noise, particularly in
                           the high-frequency domain, and also prevents plot
                           files from being too large.
    :param binafter (int): The number of low-frequency data points that are left unbinned.

    :returns (dict): Results dictionary containing
        * t : Time intervals of the trajectory
        * f : Frequency of the Fourier space components
        * V : Average system volume across the trajectory
        * P : Polarization trajectory of the solvent
        * J : Conductivity trajectory of the ionic charges
        * acf_w : Solvent polarization ACF
        * acf_iw : Polarization/conductivty cross-correlation function
        * acf_i : Ionic conductivity ACF
        * susc_w : Solvent polarization contribution to susceptibilty
        * susc_iw : Solvent/ionic contribution to susceptibility
        * susc_i : Conductivity-corrected ionic contribution to the susceptibility
        * cond : Frequency dependent conductivity
    """
    def __init__(self, ag_sol, ag_ion, temperature = 300, repair = False, 
        segs = 1, bins = 200, binafter = 20, **kwargs):
        # Initialize parent class
        super().__init__(ag_sol, ag_ion, labels = ["SOL", "ION"], restypes = ["NM", "SP"], repair = repair, **kwargs)

        self.temperature = temperature
        self.repair = repair
        self.segs = segs

        self.bins = bins
        self.binafter = binafter

    def _prepare(self):
        # Add the basic stuff from parent class
        super()._prepare()

        # Specific traits for segmenting ACFs
        self.nframes_seg = self.nframes // self.segs
        self.results["time_seg"] = self.results["time"][:self.nframes_seg]

        # Check for binning
        self.bin = not (self.nframes_seg <= self.bins)

    #==============================================================================# 
    # ACF and susceptibility calculations
    #==============================================================================# 

    def _calculate_acf_water(self):
        M = self.results["SOL_MD"]

        trj_len = self.nframes_seg * self.segs
        trjs = np.split(M[:trj_len], self.segs)

        acf = np.zeros(self.nframes_seg)
        for i in range(self.segs):
            acf += vcorrelate(trjs[i], trjs[i])
        acf *= self.pref / self.segs

        self.results["acf_w"] = acf

    def _calculate_acf_ion_water(self):
        M = self.results["SOL_MD"]
        J = self.results["ION_J"]

        trj_len = self.nframes_seg * self.segs
        trjs_M = np.split(M[:trj_len], self.segs)
        trjs_J = np.split(J[:trj_len], self.segs)

        acf = np.zeros(self.nframes_seg)
        for i in range(self.segs):
            acf += vcorrelate(trjs_M[i], trjs_J[i]) - vcorrelate(trjs_J[i], trjs_M[i])
        acf *= 0.5 * self.pref / self.segs

        self.results["acf_iw"] = acf

    def _calculate_acf_ion(self):
        J = self.results["ION_J"]

        trj_len = self.nframes_seg * self.segs
        trjs = np.split(J[:trj_len], self.segs)

        acf = np.zeros(self.nframes_seg)
        for i in range(self.segs):
            acf += vcorrelate(trjs[i], trjs[i])
        acf *= self.pref / self.segs

        self.results["acf_i"] = acf

    #==============================================================================# 

    def _calculate_susc_water(self):
        time = self.results["time_seg"]
        corr = self.results["acf_w"]
        
        # Find when to stop integrating
        idx_neg = np.argwhere(corr < 0)
        if len(idx_neg) == 0:
            idx_int = corr.size
        else:
            idx_int = idx_neg[0, 0]

        # Numerically differentiate and FT directly
        corr_int = zero_pad(corr[:idx_int], length = corr.size)
        corr_deriv = - np.append(np.diff(corr_int) / self.results["dt"], [0.0])
        
        # Do the Laplace FFT
        chi = FT(time, corr_deriv, indvar = False)
        
        # Piece together what we need
        halfN = chi.size // 2 + 1
        chi = chi[halfN:]

        self.results["susc_w"] = chi

    def _calculate_susc_ion(self):
        time = self.results["time_seg"]
        corr_iw = self.results["acf_iw"]
        corr_i = self.results["acf_i"]
        
        # Find integration cutoffs
        # acf_iw
        idx_max = np.argmax(corr_iw[:500]) # Find peak within first 500 points
        idx_negs = np.argwhere(corr_iw[idx_max:] < 0) # Find negative crossing after the peak
        if len(idx_negs) == 0:
           idx_int = corr_iw.size
        else:
           idx_int = np.min(idx_negs) + idx_max
        corr_iw_int = zero_pad(corr_iw[:idx_int], length = corr_iw.size)

        # acf_i
        t_int = 3.0 # Hard coded @ 3 ps for now
        idx_int = np.argmin(np.abs(self.results["time_seg"] - t_int))
        corr_i_int = zero_pad(corr_i[:idx_int], length = corr_i.size)

        # Calculate the susceptibility contributions
        k, chi_iw = FT(time, corr_iw_int)
        chi_i = FT(time, corr_i_int, indvar = False)

        # Determine the conductivity
        halfN = k.size // 2 + 1
        k = k[halfN:]
        sig_i = chi_i[halfN:]
        sig_iw = -1j * k * chi_iw[halfN:] 
        sig = sig_i + sig_iw

        # And scale the susceptibility
        chi_iw = -2 * chi_iw[halfN:]
        chi_i = -(1j / k) * (chi_i[halfN:] - sig[0].real)
        
        self.results["susc_iw"] = chi_iw
        self.results["susc_i"] = chi_i
        self.results["cond"] = sig

    #==============================================================================#
    # Wrapping up
    #==============================================================================#

    def _conclude(self):
        self.results["freq"] = np.fft.rfftfreq(
            self.results["time_seg"].size, d = self.results["dt"]
        )[1:-1]

        self.results["volume"] = 1e-3 * self.volume / (self.nframes + 1)
        self.results["SOL_MD"] /= 10.0
        self.results["ION_J"]  /= 10.0

        # Prefactor for autocorrelation functions
        self.pref = ec**2 * 1e9 / (3 * self.results["volume"] * kB * self.temperature * eps0)

        # Calculate the various time correlation functions needed
        if self._verbose:
            print("Calculating time correlation functions...")
        self._calculate_acf_water()
        self._calculate_acf_ion_water()
        self._calculate_acf_ion()

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
                print("Unbinned data consists of {} datapoints.".format(len(self.results["freq"])))
            self.results["freq"]    = bin_data(self.results["freq"], self.bins, after = self.binafter)
            self.results["susc_w"]  = bin_data(self.results["susc_w"], self.bins, after = self.binafter)
            self.results["susc_iw"] = bin_data(self.results["susc_iw"], self.bins, after = self.binafter)
            self.results["susc_i"]  = bin_data(self.results["susc_i"], self.bins, after = self.binafter)
            self.results["susc"]    = bin_data(self.results["susc"], self.bins, after = self.binafter)
            self.results["cond"]    = bin_data(self.results["cond"], self.bins, after = self.binafter)
        else:
            if self._verbose:
                print("Not binning data of length {}.".format(len(self.results["freq"])))

    def save(self, prefix = "", delimiter = " ", **kwargs):
        # Saves the ion trajectories and stuff
        super().save(prefix = prefix, delimiter = delimiter, **kwargs)

        # Save each of the ACFs
        acfs = np.transpose((self.results["time_seg"], self.results["acf_w"], self.results["acf_iw"], self.results["acf_i"]))
        np.savetxt(self.output + "acfs.dat", acfs, delimiter = delimiter, 
            header = "time, acf_w, acf_iw, acf_i"
        )

        # Save the susceptibilities and conductivity
        np.savetxt(self.output + "susceptibility.dat",
            np.transpose((
                self.results["freq"],
                self.results["susc"].real,    -1*self.results["susc"].imag,
                self.results["susc_w"].real,  -1*self.results["susc_w"].imag,
                self.results["susc_iw"].real, -1*self.results["susc_iw"].imag,
                self.results["susc_i"].real,  -1*self.results["susc_i"].imag
            )),
            delimiter = delimiter,
            header = "freq, susc_re, susc_im, susc_w_re, susc_w_im, susc_iw_re, susc_iw_im, susc_i_re, susc_i_im"
        )

        np.savetxt(self.output + "conductivity.dat",
            np.transpose((
                self.results["freq"],
                self.results["cond"].real, -1*self.results["cond"].imag
            )),
            delimiter = delimiter, header = "freq, cond_re, cond_im"
        )