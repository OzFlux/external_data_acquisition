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

def convert_Kelvin_to_celsius(T):

    return T - 273.15
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def convert_pressure(ps):

    return ps / 1000.0
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_Ah(Ta, q, ps):

    return get_e_from_spec_hum(q, ps) * 10**3 / ((Ta * 8.3143) / 18)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_e_from_spec_hum(q, ps):

    Md = 0.02897   # molecular weight of dry air, kg/mol
    Mv = 0.01802   # molecular weight of water vapour, kg/mol
    return q * (Md / Mv) * (ps / 1000)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_es(Ta):

    return 0.6106 * np.exp(17.27 * (Ta - 273.15) / ((Ta - 273.15)  + 237.3))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_wind_direction(u, v):

    """Returns wind direction - note that works when u and v are either
       xarray data_arrays or numpy arrays, but not for pandas dataframes"""

    wd = float(270) - np.arctan2(v, u) * float(180) / np.pi
    if isinstance(wd, pd.core.series.Series):
        return pd.Series(np.where(wd < 360, wd, wd - 360), index=wd.index,
                         name='Wd')
    return xr.where(wd < 360, wd, wd - 360)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_wind_speed(u, v):

    return np.sqrt(u**2 + v**2)
#------------------------------------------------------------------------------