#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:41:04 2020

@author: imchugh
"""

from configobj import ConfigObj
import datetime as dt
import glob
import numpy as np
import os
import pandas as pd
from pytz import timezone
import xarray as xr
import pdb

import utils

def drop_dupes(ds):
    
    idx = np.unique(ds.time.data, return_index=True)[1]
    new_ds = ds.isel(time=idx)
    if len(new_ds.time) < 2: raise RuntimeError('Too few records!')
    return new_ds

def drop_vars(ds):
    
    allowed_list = get_allowed_sub_vars()
    return_vars = [x for x in list(ds.variables) if x in allowed_list]
    return ds[return_vars]

def get_allowed_sub_vars(var=None):

    vars_list = vars_dict.keys() if not var else [var]
    master_list = []
    for var in vars_list:
        if var in no_generic_list: master_list.append(var); continue
        new_list = ([var + '_{}0'.format(str(x)) for x in range(3)] + 
                    [var + '_{}1'.format(str(x)) for x in range(3)] + 
                    [var + '_{}2'.format(str(x)) for x in range(3)])
        master_list += new_list + [x + '_QCFlag' for x in new_list]
    return master_list

def _get_generic_name(var):
        
    if 'QCFlag' in var: 
        raise RuntimeError('No defined attrs for flag variables')
    if var in ['time', 'crs']: return var
    return '_'.join(var.split('_')[:-1])    

def get_start_end_dates(current_file_str, t_offset):
    
    year_mon = os.path.splitext(f)[0].split('_')[-1]
    year_mon_py = dt.datetime.strptime(year_mon, '%Y%m')
    start = dt.datetime.combine(year_mon_py, t_offset)    
    if start.month == 12:
        year = start.year + 1
        month = 1
    else:
        year = start.year
        month = start.month + 1
    end = dt.datetime(year, month, 1, start.hour, start.minute)
    return start, end

def make_new_name(file_path, string='_new'):
    
    path, ext = os.path.splitext(file_path)
    return path + string + ext

def rename_ds(ds):
    
    q_vars = get_allowed_sub_vars('q')
    SH_vars = [x.replace('q', 'SH') for x in q_vars]
    Ah_vars = get_allowed_sub_vars('Ah')
    AH_vars = [x.replace('Ah', 'AH') for x in Ah_vars]
    rename_dict = dict(zip(q_vars + Ah_vars, SH_vars + AH_vars)) 
    return ds.rename(rename_dict)        

def set_var_attrs(ds):
    
    for var in list(ds.variables):
        try:
            generic_var = _get_generic_name(var)
            if not generic_var in no_generic_list:
                attrs = generic_dict.copy()
            else:
                attrs = {}
            attrs.update(vars_dict[generic_var])
            try:
                standard_attrs = configs['variable_attributes'][generic_var]
                if isinstance(standard_attrs['units'], list):
                    idx = unit_list_level_dict[generic_var]
                    standard_attrs['units'] = standard_attrs['units'][idx]
                attrs.update(standard_attrs)
            except KeyError:
                pass
            ds[var].attrs = attrs
        except RuntimeError:
            continue

#    if not this_var in no_generic_list:
#        this_dict = generic_dict.copy()
#    else:
#        this_dict = {}
#    try:
#        var_attrs_dict = configs['variable_attributes'][this_var]
#        if isinstance(var_attrs_dict['units'], list):
#            idx = unit_list_level_dict[var]
#            var_attrs_dict['units'] = var_attrs_dict['units'][idx]
#        this_dict.update(var_attrs_dict)
#    except KeyError: pass
#    try: this_dict.update(vars_dict[this_var])
#    except TypeError: pass
#    return this_dict

def set_var_encoding(ds):
    
    for var in list(ds.variables):
        for key in ['missing_value', '_FillValue']:
            if key in ds[var].encoding:
                ds[var].encoding[key] = -9999.0

def set_start_end_dates(ds, start, end):
    
    return ds.sel(time=slice(start, end))


#------------------------------------------------------------------------------
# DICTS
#------------------------------------------------------------------------------
generic_dict = {'height': 'not defined',
                'standard_name': 'not defined',
                'valid_range': '-1e+35,1e+35'}

vars_dict = {'AH': {'source': 'Calculated'}, 
             'Fa': {'source': 'Calculated'},
             'Fe': {'level_type': 'single', 'accum_type': 'instantaneous', 
                    'stash_code': '3234', 'grid_type': 'spatial', 
                    'source': 'access'}, 
             'Fg': {'source': 'Calculated'}, 
             'Fh': {'level_type': 'single', 'accum_type': 'instantaneous', 
                    'stash_code': '3217', 'grid_type': 'spatial',
                    'source': 'ACCESS'},
             'Fld': {'level_type': 'single', 'accum_type': 'mean',
                     'stash_code': '2207', 'accum_units': 'hrs',
                     'accum_value': '1', 'grid_type': 'spatial',
                     'source': 'ACCESS'},
             'Flu': {'source': 'Calculated'},
             'Fn': {'source': 'Calculated'},
             'Fn_lw': {'long_name': 'Net longwave radiation', 
                       'group_name': 'radiation', 'units': 'W/m^2',
                       'level_type': 'single', 'accum_type': 'mean', 
                       'stash_code': '2201', 'accum_units': 'hrs', 
                       'accum_value': '1', 'grid_type': 'spatial', 
                       'source': 'ACCESS'},
             'Fn_sw': {'long_name': 'Net shortwave radiation', 
                       'group_name': 'radiation', 'units': 'W/m^2',
                       'level_type': 'single', 'coverage_L1': '66',
                       'accum_type': 'mean', 'stash_code': '1202',
                       'accum_units': 'hrs', 'accum_value': '1',
                       'grid_type': 'spatial', 'source': 'ACCESS'},
             'Fsd': {'level_type': 'single', 'accum_type': 'mean', 
                     'stash_code': '1235', 'accum_units': 'hrs', 
                     'accum_value': '1', 'grid_type': 'spatial',
                     'source': 'ACCESS'},
             'Fsu': {'source': 'Calculated'},
             'Habl': {'level_type': 'single', 
                      'long_name': 'Planetary boundary layer height',
                      'accum_type': 'instantaneous', 'stash_code': '25', 
                      'units': 'm', 'grid_type': 'spatial', 
                      'source': 'ACCESS'},
             'Precip': {'level_type': 'single', 'accum_type': 'accumulative', 
                        'stash_code': '5226', 'accum_units': 'hrs', 
                        'accum_value': '3', 'grid_type': 'spatial',
                        'source': 'ACCESS'},
             'RH': {'source': 'Calculated'},
             'Sws': {'accum_type': 'instantaneous', 'stash_code': '8223', 
                     'level_type': 'multi', 'grid_type': 'spatial',
                     'source': 'ACCESS'},
             'Ta': {'accum_type': 'instantaneous', 'stash_code': '3236', 
                    'level_type': 'single', 'grid_type': 'spatial',
                    'source': 'ACCESS'},
             'Ts': {'accum_type': 'instantaneous', 'stash_code': '8225', 
                    'level_type': 'multi', 'grid_type': 'spatial',
                    'source': 'ACCESS'},
             'Wd': {'height': '10m', 'source': 'Calculated'},
             'Ws': {'height': '10m', 'source': 'Calculated'},
             'ps': {'accum_type': 'instantaneous', 'stash_code': '409', 
                    'level_type': 'single', 'grid_type': 'spatial',
                    'source': 'ACCESS'},
             'crs': {'grid_mapping_name': 'latitude_longitude',
                     'long_name': 'WGS 1984 datum',
                     'longitude_of_prime_meridian': '0.0',
                     'semi_major_axis': '6378137.0',
                     'inverse_flattening': '298.257223563',
                     'source': ''},
             'SH': {'level_type': 'single', 'accum_type': 'instantaneous', 
                    'stash_code': '3237', 'grid_type': 'spatial',
                    'source': 'ACCESS'},
             'time': {'long_name': 'time', 'standard_name': 'time',
                      'source': ''},
             'u': {'level_type': 'single', 'long_name': 'Wind u component',
                   'accum_type': 'instantaneous', 'stash_code': '3209', 
                   'units': 'm/s', 'height': '10m', 'grid_type': 'spatial',
                   'source': 'ACCESS'},
             'ustar': {'level_type': 'single', 'accum_type': 'instantaneous', 
                       'stash_code': '3465', 'grid_type': 'spatial', 
                       'source': 'ACCESS'},
             'v': {'level_type': 'single', 'long_name': 'Wind v component',
                   'accum_type': 'instantaneous', 'stash_code': '3210', 
                   'units': 'm/s', 'height': '10m', 'grid_type': 'spatial',
                   'source': 'ACCESS'}
             }

alias_dict = {'q': 'SH', 'Ah': 'AH'}

unit_list_level_dict = {'RH': 1, 'Ta': 0, 'Ps': 0, 'Ts': 0}

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# LISTS
#------------------------------------------------------------------------------

no_generic_list = ['time', 'crs']

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# MAIN PROGRAM
#------------------------------------------------------------------------------

cfg_path = '/mnt/OzFlux/external_data_acquisition/Code/EPCN/cfg_update.txt'
configs = ConfigObj(cfg_path)
sites = utils.get_ozflux_site_list()

for site in sites.index[11:12]:

    paths = sorted(glob.glob('/rdsi/market/access_old_site_files/monthly/*/{}*'
                             .format(site.replace(' ', ''))))
    paths = [x for x in paths if not 'old' in os.path.basename(x)]    
    tzobj = timezone(sites.loc[site, 'Time zone'])
    t_offset = tzobj.fromutc(dt.datetime(2016,6,1)).time()
    
    for f in paths:
    
        print ('Running file {}'.format(os.path.basename(f)))
        start, end = get_start_end_dates(f, t_offset)
        ds = xr.open_dataset(f)
        ds = rename_ds(ds)
        ds = drop_vars(ds)
        ds = set_start_end_dates(ds, start, end)
        try:
            ds = drop_dupes(ds)
        except RuntimeError:
            'Insufficient data in file... skipping!'
            continue
        set_var_attrs(ds)
        set_var_encoding(ds)
        new_fname = make_new_name(f)
        ds.to_netcdf(new_fname, format='NETCDF4')
        ds.close()