#!/usr/bin/env python3
# coding: utf-8

from __future__ import division
import numpy as np
import itertools

from .utils import nearest_power_two

#==============================================================================#

def correlation_1d(data1, data2 = None):
    """
    Compute the correlation of two scalar time series.
    Args:
        data1 (array-like): Input time series of shape (N,)
        data2 (array-like): Input time series of shape (N,). Defaults to copy of data1.
    Returns:
        : ndarray of shape (N,) with the correlation for
        "data1*data2[tau]" where tau is the lag in units of the timestep in the
        input data. The correlation is given from time 0 to time N.
    """
    data1 = np.asarray(data1)
    if data2 is None:
        data2 = data1
    data2 = np.asarray(data2)

    N = len(data1)
    assert N == len(data2)
    n_fft = nearest_power_two(N)

    # Pad the signal with zeros to avoid the periodic images.
    R_data1 = np.zeros(2*n_fft)
    R_data1[:N] = data1
    R_data2 = np.zeros(2*n_fft)
    R_data2[:N] = data2
    F_data1 = np.fft.fft(R_data1)
    F_data2 = np.fft.fft(R_data2)
    result = np.fft.ifft(F_data1.conj()*F_data2)

    return result[:N].real / (N - np.arange(N))

    #positive_time = result[:N].real/(N-np.arange(N))
    #negative_time = result[-N+1:][::-1].real/(N-1-np.arange(N-1))
    #return np.concatenate((negative_time[::-1], positive_time))

def correlation(data1, data2 = None):
    """Correlation between the input data using the fft algorithm.
    For D-dimensional time series, a sum is performed on the last dimension.
    Args:
        data1 (array-like): The first input signal, of shape (N,) or (N,D).
        data2 (array-like): The second input signal, of equal shape as data1. Defaults to a copy of data1.
    Returns:
        : ndarray of shape (N,) with the correlation for
        "data1*data2[tau]" where tau is the lag in units of the timestep in the
        input data. The correlation is given from time 0 to time N.
    """

    data1 = np.asarray(data1)
    if data2 is None:
        data2 = data1
    data2 = np.asarray(data2)

    if data1.shape != data2.shape:
        raise ValueError('Incompatible shapes for data1 and data2')

    if data1.ndim == 1:
        return correlation_1d(data1, data2)
    elif data1.ndim > 1:
        result = correlation_1d(data1[:,0], data2[:,0])
        for j in range(1, data1.shape[1]):
            result += correlation_1d(data1[:,j], data2[:,j])
        return result    

#==============================================================================#

def msd(pos):
    """Mean-squared displacement (MSD) of the input trajectory using the fft algorithm.
    Computes the MSD for all possible time deltas in the trajectory. The numerical results for large
    time deltas contain fewer samples than for small time times and are less accurate. This is
    intrinsic to the computation and not a limitation of the algorithm.
    Args:
        pos (array-like): The input trajectory, of shape (N,) or (N,D).
    Returns:
        : ndarray of shape (N,) with the MSD for successive linearly spaced time
        delays.
    """

    pos = np.asarray(pos)
    if pos.shape[0] == 0:
        return np.array([], dtype=pos.dtype)
    if pos.ndim==1:
        pos = pos.reshape((-1,1))
    N = len(pos)
    rsq = np.sum(pos**2, axis=1)
    MSD = np.zeros(N, dtype=float)

    SAB = correlation_1d(pos[:,0])
    for i in range(1, pos.shape[1]):
        SAB += correlation_1d(pos[:,i])

    SUMSQ = 2*np.sum(rsq)

    m = 0
    MSD[m] = SUMSQ - 2*SAB[m]*N

    MSD[1:] = (SUMSQ - np.cumsum(rsq)[:-1] - np.cumsum(rsq[1:][::-1])) / (N-1-np.arange(N-1))
    MSD[1:] -= 2*SAB[1:]

    return MSD

def cross_displacement(pos):
    """Cross displacement of the components of the input trajectory.
    Args:
        pos (array-like): The input trajectory, of shape (N, D).
    Returns:
        : list of lists of times series, where the fist two indices [i][j]
        denote the coordinates for the cross displacement: "(Delta pos[:,i]) (Delta pos[:,j])".
    """

    pos = np.asarray(pos)
    if pos.ndim != 2:
        raise ValueError("Incorrect input data for cross_displacement")
    D = pos.shape[1]

    # Precompute the component-wise MSD
    split_msd = [msd(pos_i) for pos_i in pos.T]

    # Create list of lists for the output
    result = [[] for i in range(D)]
    for i, j in itertools.product(range(D), range(D)):
        result[i].append([])

    for i, j in itertools.product(range(D), range(D)):
        if i==j:
            result[i][j] = split_msd[i]
        else:
            sum_of_pos = msd(pos[:,i]+pos[:,j])
            result[i][j] = 0.5*(sum_of_pos - split_msd[i] - split_msd[j])

    return result