#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:11:54 2020

@author: imchugh

This script takes the nearest 5 BOM stations which cover the data period for 
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
        pdb.set_trace()
        return df
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_dataset(self):

        """Collate data from individual BOM sites in xarray dataset"""

        # Get the time step and check whether to downsample
        if self.site_details['Time step'] == 30: downsample = False
        if self.site_details['Time step'] == 60: downsample = True

        # Get the dataframes for each BOM AWS site and concatenate
        lat, lon = self.site_details.Latitude, self.site_details.Longitude
        try: start = '{}0101'.format(str(int(self.site_details['Start year'])))
        except ValueError: start = None
        try: end = '{}1231'.format(str(int(self.site_details['End year'])))
        except ValueError: end = None
        nearest_stations = fbom.get_nearest_bom_station(lat, lon, start, end, 8)
        
        # Now get the data
        df_list = []
        for i, station_id in enumerate(nearest_stations.index):
            try: sub_df = self._get_dataframe(station_id)
            except FileNotFoundError: continue
            if downsample: sub_df = get_downsampled_dataframe(sub_df)
            sub_df.columns = ['{}_{}'.format(x, str(i)) for x in sub_df.columns]
            df_list.append(sub_df)
            if len(df_list) == 4: break
        df = pd.concat(df_list, axis = 1)
        df = df.loc[str(int(self.site_details['Start year'])):]
        dataset = df.to_xarray()

        # Generate variable attribute dictionaries and write to xarray dataset
        for var in df.columns:
            dataset[var].attrs = _get_var_attrs(var, nearest_stations)

        # Generate global attribute dictionaries and write to xarray dataset
        dataset.attrs = self._get_global_attrs(df)

        # Set encoding (note should be able to assign a dict to dataset.encoding
        # rather than iterating over vars but doesn't seem to work)
        dataset.time.encoding = {'units': 'days since 1800-01-01',
                                 '_FillValue': None}
        for var in dataset.data_vars:
            dataset[var].encoding = {'_FillValue': None}
        return dataset
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _get_global_attrs(self, df):

        """Make a dictionary of global attributes"""

        start_date = dt.datetime.strftime(df.index[0], '%Y-%m-%d %H:%M:%S')
        end_date = dt.datetime.strftime(df.index[-1], '%Y-%m-%d %H:%M:%S')
        run_datetime = dt.datetime.strftime(dt.datetime.now(),
                                            '%Y-%m-%d %H:%M:%S')
        return {'latitude': str(round(self.site_details.Latitude, 4)),
                'longitude': str(round(self.site_details.Longitude, 4)),
                'site_name': self.site_details.name, 'start_date': start_date,
                'end_date': end_date, 'nc_nrecs': str(len(df)),
                'nc_rundatetime': run_datetime,
                'time_step': '30', 'xl_datemode': '0'}
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

class _met_funcs(object):

    """Simple meteorological conversions"""

    def __init__(self, df):

        self.df = df

    #--------------------------------------------------------------------------
    def get_Ah(self): return (self.get_e() * 10**3 /
                              ((self.df.Ta + 273.15) * 8.3143 / 18))
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_es(self): return (0.6106 * np.exp(17.27 * self.df.Ta /
                              (self.df.Ta + 237.3)))
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_e(self): return self.get_es() * self.df.RH / 100
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_instantaneous_precip(self):

        inst_precip = self.df.Precip_accum - self.df.Precip_accum.shift()
        time_bool = self.df.local_time / 9.5 == 1
        inst_precip = inst_precip.where(~time_bool, self.df.Precip_accum)
        return inst_precip
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_q(self):

        Md = 0.02897   # molecular weight of dry air, kg/mol
        Mv = 0.01802   # molecular weight of water vapour, kg/mol
        return Mv / Md * (0.01 * self.df.RH * self.get_es() / (self.df.ps / 10))
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _us_to_from_compass(self, dctn):
        bool_idx = dctn > 90
        dctn.loc[bool_idx] = 450 - dctn.loc[bool_idx]
        dctn.loc[~bool_idx] = 90 - dctn.loc[~bool_idx]
        return dctn
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_uv_from_wswd(self):

        us_wd = self._us_to_from_compass(self.df.Wd)
        u = self.df.Ws * np.cos(np.radians(us_wd))
        v = self.df.Ws * np.sin(np.radians(us_wd))
        return u, v
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_wswd_from_uv(self):

        ws = np.sqrt(self.df.u**2 + self.df.v**2)
        wd = np.degrees(np.arctan2(self.df.v, self.df.u))
        bool_idx = wd < 0
        wd.loc[bool_idx] = wd.loc[bool_idx] + 360
        return ws, self._us_to_from_compass(wd)
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------
### FUNCTIONS ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_downsampled_dataframe(df):

    """Downsample to 1 hour"""

    met_funcs = _met_funcs(df)
    df['u'], df['v'] = met_funcs.get_uv_from_wswd()
    df.index = df.index + dt.timedelta(minutes = 30)
    rain_series = df['Precip']
    gust_series = df['Wg']
    df.drop(['Precip', 'Wd', 'Wg', 'Ws'], axis = 1, inplace = True)
    downsample_df = df.resample('60T').mean()
    met_funcs = _met_funcs(downsample_df)
    downsample_df['Ws'], downsample_df['Wd'] = met_funcs.get_wswd_from_uv()
    downsample_df.drop(['u', 'v'], axis = 1, inplace = True)
    downsample_df['Precip'] = rain_series.resample('60T').sum()
    downsample_df['Wg'] = gust_series.resample('60T').max()
    return downsample_df
#------------------------------------------------------------------------------
    
#--------------------------------------------------------------------------
def _get_var_attrs(var, nearest_stations):

    """Make a dictionary of attributes for passed variable"""

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

    generic_dict = {'instrument': '', 'valid_range': (-1e+35,1e+35),
                    'missing_value': -9999, 'height': '',
                    'standard_name': '', 'group_name': '',
                    'serial_number': ''}

    l = var.split('_')
    var, idx = l[0], int(l[1])
    var_specific_dict = vars_dict[var]
    bomsite_specific_dict = {'bom_name': nearest_stations.iloc[idx].station_name,
                             'bom_id': nearest_stations.index[idx],
                             'bom_dist (km)': str(nearest_stations.iloc[idx]['dist (km)']),
                             'time_zone': nearest_stations.iloc[idx]['time_zone']}
    return {**var_specific_dict, **bomsite_specific_dict, **generic_dict}
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
def get_instantaneous_precip(accum_precip, local_time):

    inst_precip = accum_precip - accum_precip.shift()
    time_bool = local_time / 9.5 == 1
    inst_precip = inst_precip.where(~time_bool, accum_precip)
    return inst_precip
#--------------------------------------------------------------------------   
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