import numpy as np
from scipy.optimize import curve_fit

#========================================================================================#

class FittingFunction:
    """
    Base class for performing and storing fitting results
    for an arbitrary function.

    Parameters
    ----------
    nparams : int
        Number of fitting parameters required for the model
    """
    def __init__(self, nparams):
        self.nparams = nparams
        
        self.xlim = ()
        self.is_fit = False
        self.params = np.zeros(self.nparams)
        self.param_dict = {}
        
    def _update_param_dict(self):
        pass
        
    def _func(self, x, params):
        return np.zeros(np.asarray(x).size)
    
    def evaluate(self, x):
        assert self.is_fit, "Must perform fit before evaluation."
        return self._func(x, self.params)
    
    def fit(self, x, fx, p0, nfit = None):
        """ 
        Fit the model to the provided data,
        using only the first ``nfit`` data points.
        Initial parameters ``p0`` must be of the same dimension
        as the required number of model parameters.
        """
        assert len(p0) == self.nparams, "Invalid number of initial parameters. Requires: {:d}".format(self.nparams)
        assert len(x) == len(fx), "Number of x and y data points do not match."

        if nfit is None:
            nfit = len(x)

        popt, pcov = curve_fit(lambda v, *p: self._func(v, p), x[:nfit], fx[:nfit], p0 = p0, bounds = (self.lb, self.ub))
        # Update fields
        self.xlim = (x[0], x[-1])
        self.is_fit = True
        self.params = popt
        self._update_param_dict()

#========================================================================================#
    
class MultiExponential(FittingFunction):
    """
    Fiting function for a multi-exponential function of the form
    :math:`f(x) = \sum_i A_i e^{-x / \tau_i}`

    Parameters
    ----------
    nterms : int
        Number of additive terms to include in the model
    """
    def __init__(self, nterms):
        self.nper = 2
        self.nterms = nterms
        super().__init__(self.nper*self.nterms)
        
        # Create bounds for fitting
        self.lb = np.zeros(self.nparams)
        self.ub = np.full(self.nparams, np.inf)
        
    def __repr__(self):
        return "MultiExponential(nterms = {:d}, is_fit = {:s})".format(self.nterms, str(self.is_fit))
    
    def _update_param_dict(self):
        A, tau = np.split(np.asarray(self.params), self.nper)
        self.param_dict["A"] = A
        self.param_dict["tau"] = tau
    
    def _func(self, x, params):
        A, tau = np.split(np.asarray(params), self.nper)
        fx = np.zeros(np.asarray(x).size)
        for i in range(self.nterms):
            fx += A[i]*np.exp(-x/tau[i])
        return fx

class MultiKWW(FittingFunction):
    """
    Fiting function for a multi-stretched exponential function of the form
    :math:`f(x) = \sum_i A_i ( e^{-x / \tau_i} )^{\beta_i}`

    Parameters
    ----------
    nterms : int
        Number of additive terms to include in the model
    """
    def __init__(self, nterms):
        self.nper = 3
        self.nterms = nterms
        super().__init__(self.nper*self.nterms)
        
        # Create bounds for fitting
        self.lb = np.zeros(self.nparams)
        self.ub = np.concatenate([np.full(2*self.nterms, np.inf), np.full(self.nterms, 1.0)])
        
    def __repr__(self):
        return "MultiKWW(nterms = {:d}, is_fit = {:s})".format(self.nterms, str(self.is_fit))
    
    def _update_param_dict(self):
        A, tau, beta = np.split(np.asarray(self.params), self.nper)
        self.param_dict["A"] = A
        self.param_dict["tau"] = tau
        self.param_dict["beta"] = beta
    
    def _func(self, x, params):
        A, tau, beta = np.split(np.asarray(params), self.nper)
        fx = np.zeros(np.asarray(x).size)
        for i in range(self.nterms):
            fx += A[i]*np.exp(-x/tau[i])**beta[i]
        return fx

class MultiDampedOscillator(FittingFunction):
    """
    Fiting function for a multi-damped oscillator function of the form
    :math:`f(x) = \sum_i A_i \cos(\omega_i x + \delta_i) e^{-x / \tau_i}`

    Parameters
    ----------
    nterms : int
        Number of additive terms to include in the model
    """
    def __init__(self, nterms):
        self.nper = 4
        self.nterms = nterms
        super().__init__(self.nper*self.nterms)
        
        # Create bounds for fitting
        self.lb = np.concatenate([
            np.zeros(3*self.nterms), np.full(self.nterms, -np.pi)                        
        ])
        self.ub = np.concatenate([
            np.full(3*self.nterms, np.inf), np.full(self.nterms, np.pi)   
        ])
        
    def __repr__(self):
        return "MultiDampedOscillator(nterms = {:d}, is_fit = {:s})".format(self.nterms, str(self.is_fit))
    
    def _update_param_dict(self):
        A, tau, omega, delta = np.split(np.asarray(self.params), self.nper)
        self.param_dict["A"] = A
        self.param_dict["tau"] = tau
        self.param_dict["omega"] = omega
        self.param_dict["delta"] = delta
    
    def _func(self, x, params):
        A, tau, omega, delta = np.split(np.asarray(params), 4)
        fx = np.zeros(np.asarray(x).size)
        for i in range(len(A)):
            fx += A[i]*np.cos(omega[i]*x + delta[i])*np.exp(-x/tau[i])
        return fx

class MultiColeCole(FittingFunction):
    """
    Fiting function for a multi Cole-Cole function of the form
    :math:`f(w) = \sum_i A_i/(1 + i w \tau_i)^\beta_i`

    The real and imaginary components of the function
    are fit simultaneously sharing the same parameters.
    Assumes a complex array of function values is provided for fitting.

    Parameters
    ----------
    nterms : int
        Number of additive terms to include in the model
    """
    def __init__(self, nterms):
        self.nper = 3
        self.nterms = nterms
        super().__init__(self.nper*self.nterms)
        
        # Create bounds for fitting
        self.lb = np.zeros(self.nparams)
        self.ub = np.concatenate([
            np.full(2*self.nterms, np.inf), np.ones(self.nterms)
        ])

    def _update_param_dict(self):
        A, tau, beta = np.split(np.asarray(self.params), self.nper)
        self.param_dict["A"] = A
        self.param_dict["tau"] = tau
        self.param_dict["beta"] = beta

    def _func(self, x, params):
        A, tau, beta = np.split(np.asarray(params), 3)
        fx = np.zeros(np.asarray(x).size, dtype = complex)
        for i in range(len(A)):
            fx += A[i] / (1 + 1j*x*tau[i])**beta[i]
        return fx

    def _func_both(self, x, params):
        n = len(x)//2
        fx = self._func(x[:n], params)
        both = np.zeros(2*n)
        both[:n] = fx.real
        both[n:] = fx.imag
        return both

    def fit(self, x, fx, p0, nfit = None):
        assert len(p0) == self.nparams, "Invalid number of initial parameters. Requires: {:d}".format(self.nparams)
        assert len(x) == len(fx), "Number of x and y data points do not match."
        assert fx.dtype == complex, "Must provide complex-valued function data for model."
        
        if nfit is None:
            nfit = len(x)

        xfit = np.concatenate((x[:nfit], x[:nfit]))
        ffit = np.concatenate((fx[:nfit].real, fx[:nfit].imag))
        popt, pcov = curve_fit(lambda v, *p: self._func_both(v, p), xfit, ffit, p0 = p0, bounds = (self.lb, self.ub))
        
        # Update fields
        self.xlim = (x[0], x[-1])
        self.is_fit = True
        self.params = popt
        self._update_param_dict()