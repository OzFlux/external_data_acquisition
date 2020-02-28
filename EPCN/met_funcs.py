#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:03:17 2020

Common meteorological functions

@author: imchugh
"""

import numpy as np
import pandas as pd
import xarray as xr
import pdb

#------------------------------------------------------------------------------
def convert_celsius_to_Kelvin(T):

    return T + 273.15
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def convert_Kelvin_to_celsius(T):

    return T - 273.15
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
def convert_Pa_to_kPa(ps):
    
    return ps / 1000.0
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_Ah(T, q, ps):

    """Get absolute humidity (gH2O m^-3) from temperature (K), 
       specific humidity (kgH2O kg^1) and pressure (kPa)"""
       
    return get_e_from_q(q, ps * 10**3) / ((T * 8.3143) / 18)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_e_from_q(q, ps):

    """Get vapour pressure (kPa) from specific humidity (kg kg-1) and 
       pressure (kPa)"""
    
    Md = 0.02897   # molecular weight of dry air, kg/mol
    Mv = 0.01802   # molecular weight of water vapour, kg/mol
    return q * (Md / Mv) * ps
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_es(T):

    """Get saturation vapour pressure (kPa) from temperature (K) 
       - change this to form that uses K (Clausius Clapeyron)"""
    
    return 0.6106 * np.exp(17.27 * convert_Kelvin_to_celsius(T) / 
                           (convert_Kelvin_to_celsius(T) + 237.3))
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
def get_q(RH, T, ps):

    Md = 0.02897   # molecular weight of dry air, kg/mol
    Mv = 0.01802   # molecular weight of water vapour, kg/mol
    return Mv / Md * (0.01 * RH * get_es(T) / ps)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_vectors_from_wind_dir_spd(wd, ws):

    return -ws * np.sin(np.radians(wd)), -ws * np.cos(np.radians(wd))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_wind_direction_from_vectors(u, v):

    return np.mod(270 - np.degrees(np.arctan2(v, u)), 360)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_wind_speed_from_vectors(u, v):

    return np.sqrt(u**2 + v**2)
#------------------------------------------------------------------------------