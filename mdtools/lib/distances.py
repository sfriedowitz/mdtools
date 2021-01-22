#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from MDAnalysis.lib.util import check_box
from MDAnalysis.lib.distances import _check_result_array
from MDAnalysis.lib.mdamath import triclinic_vectors

#==============================================================================#

# TODO: triclinic copies arrays, slower. Find a way to avoid...

def min_image(r, dims):
    boxtype, box = check_box(dims)
    if boxtype == "ortho":
        _min_image_ortho(r, box)
    else:
        hinv = np.linalg.inv(box)
        _min_image_triclinic(r, box, hinv)
    return r

def wrap(r, dims, img = None):
    boxtype, box = check_box(dims)
    if boxtype == "ortho":
        _wrap_ortho(r, box, img)
    else:
        hinv = np.linalg.inv(box)
        _wrap_triclinic(r, box, hinv, img)
    return r

def _min_image_ortho(r, box):
    delta = np.rint(r / box)
    r -= delta*box

def _wrap_ortho(r, box, img = None):
    delta = np.floor(r / box).astype("int")
    r -= delta*box
    if img is not None:
        img += delta

def _min_image_triclinic(r, h, hinv):
    f = np.dot(r, hinv)
    f -= np.rint(f)
    np.dot(f, h, out = r)

def _wrap_triclinic(r, h, hinv, img = None):
    f = np.dot(r, hinv)
    delta = np.floor(f).astype("int")
    f -= delta
    np.dot(f, h, out = r)
    if img is not None:
        img += delta

#==============================================================================#

def separation_array(reference, configuration, box = None):
    """
    Calculate all possible separation vectors between a reference set and another
    configuration.

    If there are ``n`` positions in `reference` and ``m`` positions in
    `configuration`, a separation array of shape ``(n, m, d)`` will be computed,
    where ``d`` is the dimensionality of each vector.

    If the optional argument `box` is supplied, the minimum image convention is
    applied when calculating separations. Either orthogonal or triclinic boxes are
    supported.
    """    
    refdim =  reference.shape[-1]
    confdim = configuration.shape[-1]
    if refdim != confdim:
        raise ValueError("Configuration dimension of {0} not equal to "
            "reference dimension of {1}".format(confdim, refdim))

    # Do the whole thing by broadcasting
    separations = reference[:, np.newaxis] - configuration
    if box is not None:
        min_image(separations, box)
    return separations