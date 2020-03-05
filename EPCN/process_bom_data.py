#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:11:54 2020

@author: imchugh

This script takes the nearest 3 BOM stations which cover the data period for 
the tower and rewrites all variables to PFP format, appending the variable names
with a number that increases with distance from the site (0 closest).

To do:
    - do timezone checking and conversion for any sites that use bom sites in
      another time zone
    - add a key to the metadata describing the bom sites, and remove from 
      individual variables
"""

#------------------------------------------------------------------------------
### MODULES (STANDARD) ###
#------------------------------------------------------------------------------

import datetime as dt
import numpy as np
import os
import pandas as pd
import xarray as xr
import sys

import pdb

#------------------------------------------------------------------------------
### MODULES (CUSTOM) ###
#------------------------------------------------------------------------------

this_path = os.path.join(os.path.dirname(__file__), '../BOM_AWS')
sys.path.append(this_path)
import bom_functions as fbom
import met_funcs
import utils

#------------------------------------------------------------------------------
### CONFIGURATIONS ###
#------------------------------------------------------------------------------

configs = utils.get_configs()
aws_file_path = configs['raw_data_write_paths']['bom'] 
nc_file_path = configs['nc_data_write_paths']['bom']

#------------------------------------------------------------------------------
### CLASSES ###
#------------------------------------------------------------------------------

class bom_data_converter(object):

    """Converts the raw text files into dataframe, xarray Dataset and
       netCDF"""

    def __init__(self, site_details):

        self.site_details = site_details

    #--------------------------------------------------------------------------
    def _get_dataframe(self, station_id):

        """Make a dataframe and convert data to appropriate units"""

        fname = os.path.join(aws_file_path, 'HM01X_Data_{}.txt'.format(station_id))
        df = pd.read_csv(fname, low_memory = False)
        new_cols = (df.columns[:5].tolist() +
                    ['hour_local', 'minute_local', 'year', 'month', 'day',
                     'hour', 'minute'] + df.columns[12:].tolist())
        df.columns = new_cols
        df.index = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
        df.index.name = 'time'
        keep_cols = [12, 14, 16, 18, 20, 22, 24, 26]
        parse_cols = ['hour_local', 'minute_local'] + df.columns[keep_cols].tolist()
        for var in parse_cols: df[var] = pd.to_numeric(df[var], errors='coerce')
        local_time = df['hour_local'] + df['minute_local'] / 60
        df = df.iloc[:, keep_cols]
        df.columns = ['Precip_accum', 'Ta', 'Td', 'RH', 'Ws', 'Wd', 'Wg', 'ps']
        df['Precip'] = get_instantaneous_precip(df.Precip_accum, local_time)
        df.drop('Precip_accum', axis=1, inplace=True)
        df.ps = df.ps / 10 # Convert hPa to kPa
        T_K = met_funcs.convert_celsius_to_Kelvin(df.Ta) # Get Ta in K
        df['q'] = met_funcs.get_q(df.RH, T_K, df.ps)
        df['Ah'] = met_funcs.get_Ah(T_K, df.q, df.ps)
        _interpolate_missing(df)
        if self.site_details['Time step'] == 60: df = _resample_dataframe(df)
        _apply_range_limits(df)
        return df
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_dataset(self):

        """Collate data from individual BOM sites in xarray dataset"""

        # Get required args to find nearest stations
        lat, lon = self.site_details.Latitude, self.site_details.Longitude
        try: start = '{}0101'.format(str(int(self.site_details['Start year'])))
        except ValueError: start = None
        try: end = '{}1231'.format(str(int(self.site_details['End year'])))
        except ValueError: end = None
        nearest_stations = fbom.get_nearest_bom_station(lat, lon, start, end, 5)
        
        # Now get the data and combine
        df_list = []
        for i, station_id in enumerate(nearest_stations.index):
            try: sub_df = self._get_dataframe(station_id)
            except FileNotFoundError: continue
            sub_df.columns = ['{}_{}'.format(x, str(i)) for x in sub_df.columns]
            df_list.append(sub_df)
            if i == 3: break
        df = pd.concat(df_list, axis = 1)
        df = df.loc[str(int(self.site_details['Start year'])):]
        ds = df.to_xarray()
        ds = ds.fillna(-9999.0)

        # Generate variable and global attribute dictionaries and write to xarray dataset
        _set_var_attrs(ds, nearest_stations)
        ds = xr.merge([ds, make_qc_flags(ds)])
        _set_global_attrs(ds, self.site_details)

        return ds
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    def get_nearest_stations(self):
        
        return fbom.get_nearest_bom_station(self.site_details.Latitude, 
                                            self.site_details.Longitude)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def write_to_netcdf(self, write_path):

        site_name = self.site_details.name
        print ('Writing netCDF file for site {}'.format(site_name))
        dataset = self.get_dataset()
        fname = '{}_AWS.nc'.format(''.join(site_name.split(' ')))
        target = os.path.join(write_path, fname)
        dataset.to_netcdf(target, format='NETCDF4')
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------
### FUNCTIONS ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _apply_range_limits(df):

    """Screen implausible data from dataframe"""
    
    for var in df.columns:
        lims = range_dict[var]
        df[var] = df[var].where(cond=(df[var] >= lims[0]) & (df[var] <= lims[1]))
    df.Precip.where(~((df.Precip >= -0.01) & (df.Precip <= 0.01)), 0, inplace=True)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _interpolate_missing(df):
    
    """Interpolate missing data where gap is less than 2 hours"""
    
    # Note that we don't do interpolation on precip data, which has been dealt with
    df['u'], df['v'] = met_funcs.get_uv_from_wdws(df['Wd'],df['Ws'])
    var_list = df.columns.tolist()
    var_list.remove('Precip')
    for var in var_list: df[var].interpolate(limit=4, inplace=True)
    df['Ws'] = met_funcs.get_ws_from_uv(df['u'], df['v'])
    df.drop(['u', 'v'], axis=1, inplace=True)
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
def _resample_dataframe(df):

    """Downsample dataframe to 1 hour"""
    df['u'], df['v'] = met_funcs.get_uv_from_wdws(df['Wd'], df['Ws'])
    df.index = df.index + dt.timedelta(minutes = 30)
    rain_series = df['Precip']
    gust_series = df['Wg']
    df.drop(['Precip', 'Wd', 'Wg', 'Ws'], axis = 1, inplace = True)
    downsample_df = df.resample('60T').mean()
    downsample_df['Ws'] = met_funcs.get_ws_from_uv(df['u'], df['v'])
    downsample_df['Wd'] = (
        met_funcs.get_wd_from_uv(df['u'], df['v']))
    downsample_df.drop(['u', 'v'], axis = 1, inplace = True)
    downsample_df['Precip'] = rain_series.resample('60T').sum()
    downsample_df['Wg'] = gust_series.resample('60T').max()
    return downsample_df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_instantaneous_precip(precip_accum, local_time):

    precip_accum_interp = precip_accum.interpolate(limit=4)
    precip_inst = precip_accum_interp - precip_accum_interp.shift()
    reset_bool = local_time / 9.5 == 1
    precip_inst.where(~reset_bool, precip_accum_interp, inplace=True)
    neg_bool = (local_time / 9 == 1) & (precip_inst < 0)
    precip_inst.where(~neg_bool, 0, inplace=True)
    return precip_inst
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def make_qc_flags(ds):

    """Generate QC flags for all variables in the ds"""

    da_list = []
    for var in ds.variables:
        if var in ds.dims: continue
        da = xr.where(~np.isnan(ds[var]), 0, 10)
        da.name = var + '_QCFlag'
        da_list.append(da)
    return xr.merge(da_list)
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
def _set_global_attrs(ds, site_details):

    """Make a dictionary of global attributes"""

    ds.attrs = {'nc_nrecs': len(ds.time),
                'start_date': (dt.datetime.strftime
                               (pd.Timestamp(ds.time[0].item()),
                                '%Y-%m-%d %H:%M:%S')),
                'end_date': (dt.datetime.strftime
                             (pd.Timestamp(ds.time[-1].item()),
                              '%Y-%m-%d %H:%M:%S')),
                'latitude': site_details.Latitude,
                'longitude': site_details.Longitude,
                'elevation': site_details.Altitude,
                'site_name': site_details.name,
                'time_step': site_details['Time step'],
                'nc_rundatetime': dt.datetime.strftime(dt.datetime.now(),
                                                       '%Y-%m-%d %H:%M:%S'),
                'xl_datemode': '0'}

    ds.time.encoding = {'units': 'days since 1800-01-01',
                        '_FillValue': None}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _set_var_attrs(ds, nearest_stations):

    """Make a dictionary of attributes for passed variable"""

    def get_bom_dict(idx):
            
        return {'bom_name': nearest_stations.iloc[idx].station_name,
                'bom_id': nearest_stations.index[idx],
                'bom_dist (km)': str(nearest_stations.iloc[idx]['dist (km)']),
                'time_zone': nearest_stations.iloc[idx]['time_zone']}
            
    vars_dict = {'Ah': {'long_name': 'Absolute humidity',
                        'units': 'g/m3'},
                 'Precip': {'long_name': 'Precipitation total over time step',
                            'units': 'mm/30minutes'},
                 'ps': {'long_name': 'Air pressure',
                        'units': 'kPa'},
                 'q': {'long_name': 'Specific humidity',
                       'units': 'kg/kg'},
                 'RH': {'long_name': 'Relative humidity',
                        'units': '%'},
                 'Ta': {'long_name': 'Air temperature',
                        'units': 'C'},
                 'Td': {'long_name': 'Dew point temperature',
                        'units': 'C'},
                 'Wd': {'long_name': 'Wind direction',
                        'units': 'degT'},
                 'Ws': {'long_name': 'Wind speed',
                        'units': 'm/s'},
                 'Wg': {'long_name': 'Wind gust',
                        'units': 'm/s'}}

    generic_dict = {'instrument': '', 'valid_range': '-1e+35,1e+35',
                    'missing_value': -9999, 'height': '',
                    'standard_name': '', 'group_name': '',
                    'serial_number': ''}
    
    for this_var in list(ds.variables):
        if this_var in ds.dims: continue
        l = this_var.split('_')
        var, idx = l[0], int(l[1])
        var_specific_dict = vars_dict[var]
        bom_dict = get_bom_dict(idx)
        ds[this_var].attrs = {**var_specific_dict, **bom_dict, **generic_dict}
        ds[this_var].encoding = {'_FillValue': -9999}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### GLOBALS ###
#------------------------------------------------------------------------------
        
#------------------------------------------------------------------------------
range_dict = {'Precip': [-0.01, 100], 
              'Ta': [-40, 60], 
              'Td': [-60, 60],
              'RH': [0, 100],
              'Ws': [0, 100],
              'Wg': [0, 100],
              'Wd': [0, 360], 
              'ps': [70, 110],
              'q': [0, 1],
              'Ah': [0, 80]}
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
### MAIN PROGRAM
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Get conversion class and write text data to nc file
    sites = utils.get_ozflux_site_list()
    for site in sites.index:
        conv_class = bom_data_converter(sites.loc[site])
        conv_class.write_to_netcdf(nc_file_path)
#------------------------------------------------------------------------------