#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import re
import numpy as np

#==============================================================================#

def atomgroup_header(atomgroup):
    """
    Return a string containing info about the AtomGroup 
    containing the total number of atoms, 
    the including residues and the number of residues.
    Useful for writing output file headers.
    """
    unq_res, n_unq_res = np.unique(
        atomgroup.residues.resnames, return_counts=True)
    return "{} atom(s): {}".format(
        atomgroup.n_atoms, ", ".join(
            "{} {}".format(*i) for i in np.vstack([n_unq_res, unq_res]).T))

def fill_template(template, vars, s = "<", e = ">"):
    """
    Search and replace tool for filling template files.
    Replaces text bounded by the delimiters `s` and `e`
    with values found in the lookup dictionary `vars`.
    """
    exp = s + "\w*" + e
    matches = re.findall(exp, template)
    for m in matches:
        key = m[1:-1]
        template = template.replace(m, str(vars.get(key, m)))
    return template

def save_path(prefix = ""):
    """Returns a formatted output location for a given file prefix."""
    if prefix != "" and prefix[-1] != "/":
        prefix += "_"
    output = prefix if os.path.dirname(prefix) else os.path.join(os.getcwd(), prefix)
    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(prefix))

    return output

#==============================================================================#

def nearest_power_two(n):
    """
    Select the closest i such that n<=2**i.
    """
    current_exp = int(np.ceil(np.log2(n+1)))
    if n == 2**current_exp:
        n_fft = n
    if n < 2**current_exp:
        n_fft = 2**current_exp
    elif n > 2**current_exp:
        n_fft = 2**(current_exp+1)

    return n_fft    

def zero_pad(x, n):
    """
    Pad an array to length `n` with zeros.
    If the original array length is greater than `n`,
    a copy of the original array is returned with it's length unchanged.
    """
    nx = len(x)
    if n < nx:
        n = nx
    new = np.zeros((n, *x.shape[1:]), dtype = x.dtype)
    new[:nx] = x

    return new

def bin_data(arr, nbins, after = 1, log = True):
    """
    Averages array values in bins for easier plotting.
    """
    # Determine indices to average between
    if log:
        bins = np.logspace(np.log10(after), np.log10(len(arr)-1), nbins+1).astype(int)
    else:
        bins = np.linspace(after, len(arr), nbins+1).astype(int)
    bins = np.unique(np.append(np.arange(after), bins))

    avg = np.zeros(len(bins)-1, dtype = arr.dtype)
    for i in range(len(bins)-1):
        avg[i] = np.mean(arr[bins[i]:bins[i+1]])
        
    return avg