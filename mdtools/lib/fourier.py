#!/usr/bin/env python3
# coding: utf-8

import numpy as np

#==============================================================================#

delta_tol = 1e-8  # Max variation from the mean dt or dk that is allowed (~1e-10 suggested)

def kfft(t):
    n = len(t)
    dt = (t[-1] - t[0]) / float(n - 1) 
    k = np.fft.fftshift(np.fft.fftfreq(n, d = dt) * 2 * np.pi)
    return k

def fft_signal(t, x, indvar = True):
    """
    Discrete fast fourier transform.
    Takes the time series and the function as arguments.
    By default, returns the FT and the frequency.
    Setting indvar = False returns only the FT values.
    """
    n = len(t)
    a, b = np.min(t), np.max(t)

    dt = (t[-1] - t[0]) / float(n - 1)  # Timestep
    if (abs((t[1:] - t[:-1] - dt)) > delta_tol).any():
        raise RuntimeError("Time series not equally spaced!")

    # Calculate frequency values for FT
    k = np.fft.fftshift(np.fft.fftfreq(n, d = dt) * 2 * np.pi)
    # Calculate FT of data
    xf = np.fft.fftshift(np.fft.fft(x))
    xf *= dt * np.exp(-1j * k * a) # Scales the FT by timestep and phase factor

    return (k, xf) if indvar else xf

def ifft_signal(k, xf, indvar = True):
    """
    Inverse discrete fast fourier transform.
    Takes the frequency series and the function as arguments.
    By default, returns the iFT and the time series,
    Setting indvar = False means the function returns only the inverse FT values.
    """
    n = len(k)

    dk = (k[-1] - k[0]) / float(n - 1)  # Freq spacing
    if (abs((k[1:] - k[:-1] - dk)) > delta_tol).any():
        raise RuntimeError("Time series not equally spaced!")

    # Calculate time values for iFT
    t = np.fft.ifftshift(np.fft.fftfreq(n, d = dk) * 2 * np.pi) 
    # Calculate iFT of data
    x = np.fft.ifftshift(np.fft.ifft(xf))
    x *= n * dk / (2 * np.pi)

    if n % 2 == 0:
        x *= np.exp(-1j * t * n * dk / 2)
    else:
        x *= np.exp(-1j * t * (n - 1) * dk / 2.)

    return (t, x) if indvar else x

#==============================================================================#

# Evaluting fourier integrals using Filon's trapezoidal rule

def _filon_wtn(theta):
    # From theta, calculate wt and wn
    wn = np.zeros(theta.size, dtype = complex)

    # theta == 0 needs special treatment
    inz, = theta.nonzero()
    iz,  = np.where(theta == 0.0)
    if len(iz) > 0:
        wn[iz] = 0.5
        theta = theta[inz]

    wn[inz] = (1.0 + 1j*theta - np.exp(1j*theta)) / theta**2
    wt = wn + np.conj(wn)
    
    return wt, wn

def _filon_abg(theta):
    # From theta, calculate alpha, beta, and gamma
    alpha = np.zeros(theta.size)
    beta = np.zeros(theta.size)
    gamma = np.zeros(theta.size)

    # theta==0 needs special treatment
    inz, = theta.nonzero()
    iz, = np.where(theta == 0.0)
    if len(iz) > 0:
        beta[iz] = 2.0 / 3.0
        gamma[iz] = 4.0 / 3.0
        theta = theta[inz]

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin2_t = sin_t * sin_t
    cos2_t = cos_t * cos_t
    theta2 = theta * theta
    itheta3 = 1.0 / (theta2 * theta)

    alpha[inz] = itheta3 * (theta2 + theta * sin_t * cos_t - 2 * sin2_t)
    beta[inz] = 2 * itheta3 * (theta * (1 + cos2_t) - 2 * sin_t * cos_t)
    gamma[inz] = 4 * itheta3 * (sin_t - theta * cos_t)

    return alpha, beta, gamma

def _filon_sin(x, f, k, dx):       
    # Get the weights
    theta = dx*k
    alpha, beta, gamma = _filon_abg(theta)
    
    # Compute the integral
    scx = np.sin(np.outer(k, x))    
    scx[:,0] *= 0.5
    scx[:,-1] *= 0.5 
    
    I = alpha * (f[0]*np.cos(k*x[0]) - f[-1]*np.cos(k*x[-1]))
    I += beta * np.sum(f[::2] * scx[:,::2], axis = -1) + gamma * np.sum(f[1::2]*scx[:,1::2], axis = -1)
    I *= dx
    return I

def _filon_cos(x, f, k, dx):
    N = len(x)
    a, b = np.min(x), np.max(x)
     
    # Get the weights
    theta = dx*k
    alpha, beta, gamma = _filon_abg(theta)
    
    # Compute the integral
    scx = np.cos(np.outer(k, x))    
    scx[:,0] *= 0.5
    scx[:,-1] *= 0.5 
    
    I = alpha*(f[0]*np.sin(k*x[0]) - f[-1]*np.sin(k*x[-1]))
    I += beta*np.sum(f[::2] * scx[:,::2], axis = -1) + gamma*np.sum(f[1::2]*scx[:,1::2], axis = -1)
    I *= dx
    return I

def _filon_fft(x, f, k, dx):       
    # Get the weights
    theta = dx*k
    wt, wn = _filon_wtn(theta)

    # Compute the integral
    I = wt*np.fft.fftshift(np.fft.fft(f)) + wn*(f[-1]*np.exp(1j*k*x[-1]) - f[0]*np.exp(1j*k*x[0]))
    I *= dx
    
    return I

def sin_integrate(x, f, k = None, nk = None, indvar = True):
    """Calculate the integral
    :math:`\int_{x_0}^{2n\Delta x} f(x) \sin(k x) dx`.

    Parameters
    ----------
    x : array
        values of x-axis to integrate along
    f : float
        functions values along x-axis
    k : array
        values of reciprocal axis at which to evaluate transform;
        if ``k`` is not provided, ``2*pi*linspace(0.0, 1/(2*dx), nk)``,
        will be used.
    nk : int
        number of ``k`` values to use if not provided, default to ``len(x)``
    kzero: bool
        whether to return ``k`` containing zero frequency component,
        default to False

    Returns
    -------
    float
        tuple of ``k`` and ``F`` values
    """
    N = len(x)
    a, b = np.min(x), np.max(x)
    dx = (x[-1] - x[0]) / float(N - 1)
    
    if k is None:
        nk = nk if nk is not None else N
        k = 2 * np.pi * np.linspace(0.0, 1.0/(2*dx), nk)
        
    ff = _filon_sin(x, f, k, dx)
    return (k, ff) if indvar else ff

def cos_integrate(x, f, k = None, nk = None, indvar = True):
    """Calculate the integral
    :math:`\int_{x_0}^{2n\Delta x} f(x) \cos(k x) dx`.

    Parameters
    ----------
    x : array
        values of x-axis to integrate along
    f : float
        functions values along x-axis
    k : array
        values of reciprocal axis at which to evaluate transform;
        if ``k`` is not provided, ``2*pi*linspace(0.0, 1/(2*dx), nk)``,
        will be used.
    nk : int
        number of ``k`` values to use if not provided, default to ``len(x)``
    kzero: bool
        whether to return ``k`` containing zero frequency component,
        default to False

    Returns
    -------
    float
        tuple of ``k`` and ``F`` values
    """
    N = len(x)
    a, b = np.min(x), np.max(x)
    dx = (x[-1] - x[0]) / float(N - 1)
    
    if k is None:
        nk = nk if nk is not None else N
        k = 2 * np.pi * np.linspace(0.0, 1.0/(2*dx), nk)
        
    ff = _filon_cos(x, f, k, dx)
    return (k, ff) if indvar else ff

def fourier_integrate(x, f, k = None, nk = None, fft = True, indvar = True):
    """Calculate the integral
    :math:`\int_{x_0}^{2n\Delta x} f(x) exp(-1 i k x) dx`.

    Parameters
    ----------
    x : array
        values of x-axis to integrate along
    f : float
        functions values along x-axis
    k : array
        values of reciprocal axis at which to evaluate transform;
        if ``k`` is not provided, ``2*pi*linspace(0.0, 1/(2*dx), nk)``,
        will be used.
    nk : int
        number of ``k`` values to use if not provided, default to ``len(x)``
    fft: bool
        whether to use Filon's method via FFT, or sin/cos transforms.
        If True, ``k`` values are automatically generated.

    Returns
    -------
    float
        tuple of ``k`` and ``F`` values
    """
    N = len(x)
    a, b = np.min(x), np.max(x)
    dx = (x[-1] - x[0]) / float(N - 1)
    
    if fft:
        k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d = dx))
        ff = _filon_fft(x, f, k, dx)
    else:
        if k is None:
            nk = nk if nk is not None else N
            k = 2 * np.pi * np.linspace(0.0, 1.0/(2*dx), nk)
            
        ff = np.zeros(k.size, dtype = complex)
        ff.real = _filon_cos(x, f, k, dx)
        ff.imag = -1 * _filon_sin(x, f, k, dx)
    
    return (k, ff) if indvar else ff