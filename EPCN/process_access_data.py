#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:16:21 2019

@author: ian
"""

#------------------------------------------------------------------------------
### MODULES (STANDARD) ###
#------------------------------------------------------------------------------

import datetime as dt
import glob
import numpy as np
import os
import pandas as pd
from pytz import timezone
import xarray as xr
import pdb

#------------------------------------------------------------------------------
### MODULES (CUSTOM) ###
#------------------------------------------------------------------------------

import utils

#------------------------------------------------------------------------------
### CONFIGURATIONS ###
#------------------------------------------------------------------------------

configs = utils.get_configs()
access_file_path = configs['nc_data_write_paths']['access']

#------------------------------------------------------------------------------
### CLASSES ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class access_data_converter():

    def __init__(self, site_details, return_soil_depths=False):

        self.site_name = site_details.name
        self.latitude = round(site_details.Latitude, 4)
        self.longitude = round(site_details.Longitude, 4)
        self.time_step = site_details['Time step']
        self.time_zone = site_details['Time zone']
        self.return_soil_depths = return_soil_depths
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def create_dataset(self):

        in_ds = self.get_raw_file()
        results = []
        for i, this_lat in enumerate(in_ds.lat):
            for j, this_lon in enumerate(in_ds.lon):
                l = []
                for var in vars_dict.keys():
                    l += _get_data(in_ds, var, this_lat, this_lon,
                                   self.return_soil_depths)
                df = pd.concat(l, axis=1)
                df.index = in_ds.time.data
                df.index.name = 'time'
                df = df.loc[~df.index.duplicated()]
                pdb.set_trace()
                conv_ds = do_conversions(df).to_xarray()
                _set_variable_attributes(conv_ds,
                                         round(this_lat.item(), 4),
                                         round(this_lon.item(), 4))
                results.append(_rename_variables(conv_ds, i, j))
        in_ds.close()
        merge_ds = xr.merge(results, compat='override')
        offset = self.get_utc_offset()
        merge_ds.time.data = (pd.to_datetime(merge_ds.time.data) + 
                              dt.timedelta(hours=offset))
        if self.time_step == 30: out_ds = _resample_dataset(merge_ds)
        else: out_ds = merge_ds
        self._set_global_attributes(out_ds)
        return out_ds
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_file_list(self):

        search_str = self.site_name.replace(' ', '')
        return sorted(glob.glob(access_file_path +
                                '/Monthly_files/**/{}*'.format(search_str)))
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_raw_file(self):

        def preproc(ds):
            idx = np.unique(ds.time.data, return_index=True)[1]
            return ds.isel(time=idx)
        
        return xr.open_mfdataset(self.get_file_list(), combine='by_coords', 
                                 preprocess=preproc)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_utc_offset(self):
        
        tz_obj = timezone(self.time_zone)
        now_time = dt.datetime.now()
        return (tz_obj.utcoffset(now_time) - tz_obj.dst(now_time)).seconds / 3600
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    def _set_global_attributes(self, ds):

        ds.attrs = {'nrecs': len(ds.time),
                    'start_date': (dt.datetime.strftime
                                   (pd.Timestamp(ds.time[0].item()),
                                    '%Y-%m-%d %H:%M:%S')),
                    'end_date': (dt.datetime.strftime
                                 (pd.Timestamp(ds.time[-1].item()),
                                  '%Y-%m-%d %H:%M:%S')),
                    'latitude': self.latitude,
                    'longitude': self.longitude,
                    'site_name': self.site_name,
                    'time_step': self.time_step,
                    'xl_datemode': 0}
        ds.time.encoding = {'units': 'days since 1800-01-01',
                            '_FillValue': None}
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def write_to_netcdf(self, write_path):

        print('Writing netCDF file for site {}'.format(self.site_name))
        dataset = self.create_dataset()
        fname = '{}_ACCESS.nc'.format(''.join(self.site_name.split(' ')))
        target = os.path.join(write_path, 'PFP_format', fname)
        dataset.to_netcdf(target, format='NETCDF4')
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### FUNCTIONS ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_data(ds, var, lat, lon, return_soil_depths):

    swap_var = vars_dict[var]
    if len(ds[var].dims) == 3:
        return [pd.Series(_screen_vars(ds[var].sel(lat=lat, lon=lon)),
                          name=swap_var)]
    if return_soil_depths:
        l = []
        for this_dim in ds.soil_lvl:
            name = '{}_{}m'.format(swap_var, str(round(this_dim.item(), 2)))
            l.append(pd.Series(_screen_vars(ds[var].sel(soil_lvl=this_dim,
                                                        lat=lat, lon=lon)),
                               name=name))
        return l
    return [pd.Series(_screen_vars(ds[var].sel(soil_lvl=ds.soil_lvl[0],
                                               lat=lat, lon=lon)),
                      name=swap_var)]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_var_attrs(var):

    """Make a dictionary of attributes for passed variable"""

    generic_dict = {'instrument': '', 'valid_range': (-1e+35,1e+35),
                    'missing_value': -9999, 'height': '',
                    'standard_name': '', 'group_name': '',
                    'serial_number': ''}

    generic_dict.update(attrs_dict[var])
    return generic_dict
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _rename_variables(ds, i, j):

    var_list = [x for x in list(ds.variables) if not x in list(ds.dims)]
    name_swap_dict = {x: '{}_{}'.format(x, str(i) + str(j))
                      for x in var_list}
    return ds.rename(name_swap_dict)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _resample_dataset(ds):

    precip_list = [x for x in ds if 'Precip' in x]
    no_precip_list = [x for x in ds if not 'Precip' in x]
    new_ds = ds[no_precip_list].resample(time='30T').interpolate('linear')
    for var in precip_list:
        cuml_precip = ds[var].cumsum()
        cuml_precip = cuml_precip.resample(time='30T').interpolate('linear')
        new_ds[var] = cuml_precip - cuml_precip.shift(time=1)
    return new_ds
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _screen_vars(series):

    range_lims = range_dict[series.name]
    return series.where((series >= range_lims[0]) & (series <= range_lims[1]))
#------------------------------------------------------------------------------

#--------------------------------------------------------------------------
def _set_variable_attributes(ds, latitude, longitude):

    for this_var in list(ds.variables):
        if this_var == 'time': continue
        try:
            var_attrs = _get_var_attrs(this_var)
        except KeyError:
            var_attrs = _get_var_attrs(this_var.split('_')[0])
        var_attrs.update({'latitude': latitude,
                          'longitude': longitude})
        ds[this_var].attrs = var_attrs
        ds[this_var].encoding = {'_FillValue': -9999}
#--------------------------------------------------------------------------

#------------------------------------------------------------------------------
### CONVERSION FUNCTIONS ###
#------------------------------------------------------------------------------

def convert_Kelvin_to_celsius(s):

    return s - 273.15
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def convert_pressure(df):

    return df.ps / 1000.0
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_Ah(df):

    return get_e(df) * 10**3 / ((df.Ta * 8.3143) / 18)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_e(df):

    Md = 0.02897   # molecular weight of dry air, kg/mol
    Mv = 0.01802   # molecular weight of water vapour, kg/mol
    return df.q * (Md / Mv) * (df.ps / 1000)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_energy_components(df):

    new_df = pd.DataFrame(index=df.index)
    new_df['Fsu'] = df.Fsd - df.Fn_sw
    new_df['Flu'] = df.Fld - df.Fn_lw
    new_df['Fn'] = (df.Fsd - new_df.Fsu) + (df.Fld - new_df.Flu)
    new_df['Fa'] = df.Fh + df.Fe
    new_df['Fg'] = new_df.Fn - new_df.Fa
    return new_df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_es(df):

    return 0.6106 * np.exp(17.27 * (df.Ta - 273.15) / ((df.Ta - 273.15)  + 237.3))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_Rh(df):

    return get_e(df) / get_es(df) * 100
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_wind_direction(df):

    s = float(270) - np.arctan2(df.v, df.u) * float(180) / np.pi
    s.loc[s > 360] -= float(360)
    return s
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_wind_speed(df):

    return np.sqrt(df.u**2 + df.v**2)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def do_conversions(df):

    Sws_list = list(filter(lambda x: 'Sws' in x, df.columns))
    Ts_list = list(filter(lambda x: 'Ts' in x, df.columns))
    full_list = ['Habl', 'Fsd', 'Fld', 'Fh', 'Fe', 'Precip', 'q', 'u', 'v'] + Sws_list
    new_df = df[full_list].copy()
    for var in Ts_list: new_df[var] = convert_Kelvin_to_celsius(df[var])
    new_df['Ta'] = convert_Kelvin_to_celsius(df.Ta)
    new_df['RH'] = get_Rh(df)
    new_df['Ah'] = get_Ah(df)
    new_df['Ws'] = get_wind_speed(df)
    new_df['Wd'] = get_wind_direction(df)
    new_df['ps'] = convert_pressure(df)
    new_df = new_df.join(get_energy_components(df))
    return new_df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### GLOBALS ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
attrs_dict = {'Ah': {'long_name': 'Absolute humidity',
                     'units': 'g/m3'},
              'Fa': {'long_name': 'Calculated available energy',
                     'units': 'W/m2'},
              'Fe': {'long_name': 'Surface latent heat flux',
                     'units': 'W/m2'},
              'Fg': {'long_name': 'Calculated ground heat flux',
                     'units': 'W/m2',
                     'standard_name': 'downward_heat_flux_in_soil'},
              'Fh': {'long_name': 'Surface sensible heat flux',
                     'units': 'W/m2'},
              'Fld': {'long_name':
                      'Average downward longwave radiation at the surface',
                      'units': 'W/m2'},
              'Flu': {'long_name':
                      'Average upward longwave radiation at the surface',
                      'standard_name': 'surface_upwelling_longwave_flux_in_air',
                      'units': 'W/m2'},
              'Fn': {'long_name': 'Calculated net radiation',
                     'standard_name': 'surface_net_allwave_radiation',
                     'units': 'W/m2'},
              'Fsd': {'long_name': 'average downwards shortwave radiation at the surface',
                      'units': 'W/m2'},
              'Fsu': {'long_name': 'average upwards shortwave radiation at the surface',
                      'standard_name': 'surface_upwelling_shortwave_flux_in_air',
                      'units': 'W/m2'},
              'Habl': {'long_name': 'planetary boundary layer height',
                       'units': 'm'},
              'Precip': {'long_name': 'Precipitation total over time step',
                         'units': 'mm/30minutes'},
              'ps': {'long_name': 'Air pressure',
                     'units': 'kPa'},
              'q': {'long_name': 'Specific humidity',
                    'units': 'kg/kg'},
              'RH': {'long_name': 'Relative humidity',
                     'units': '%'},
              'Sws': {'long_name': 'soil_moisture_content', 'units': 'frac'},
              'Ta': {'long_name': 'Air temperature',
                     'units': 'C'},
              'Ts': {'long_name': 'soil temperature',
                     'units': 'C'},
              'u': {'long_name': '10m wind u component',
                    'units': 'm s-1'},
              'v': {'long_name': '10m wind v component',
                    'units': 'm s-1'},
              'Wd': {'long_name': 'Wind direction',
                     'units': 'degT'},
              'Ws': {'long_name': 'Wind speed',
                     'units': 'm/s'}}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
funcs_dict = {'av_swsfcdown': [0, 1400],
              'av_netswsfc': [0, 1400],
              'av_lwsfcdown': [200, 600],
              'av_netlwsfc': [-300, 300],
              'temp_scrn': [230, 330],
              'qsair_scrn': [0, 1],
              'soil_mois': [0, 100],
              'soil_temp': [210, 350],
              'u10': [-50, 50],
              'v10': [-50, 50],
              'sfc_pres': [75000, 110000],
              'inst_prcp': [0, 100],
              'sens_hflx': [-200, 1000],
              'lat_hflx': [-200, 1000],
              'abl_ht': [0, 5000]}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
range_dict = {'av_swsfcdown': [0, 1400],
              'av_netswsfc': [0, 1400],
              'av_lwsfcdown': [200, 600],
              'av_netlwsfc': [-300, 300],
              'temp_scrn': [230, 330],
              'qsair_scrn': [0, 1],
              'soil_mois': [0, 100],
              'soil_temp': [210, 350],
              'u10': [-50, 50],
              'v10': [-50, 50],
              'sfc_pres': [75000, 110000],
              'inst_prcp': [0, 100],
              'sens_hflx': [-200, 1000],
              'lat_hflx': [-200, 1000],
              'abl_ht': [0, 5000]}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
vars_dict = {'av_swsfcdown': 'Fsd',
             'av_netswsfc': 'Fn_sw',
             'av_lwsfcdown': 'Fld',
             'av_netlwsfc': 'Fn_lw',
             'temp_scrn': 'Ta',
             'qsair_scrn': 'q',
             'soil_mois': 'Sws',
             'soil_temp': 'Ts',
             'u10': 'u',
             'v10': 'v',
             'sfc_pres': 'ps',
             'inst_prcp': 'Precip',
             'sens_hflx': 'Fh',
             'lat_hflx': 'Fe',
             'abl_ht': 'Habl'}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### MAIN PROGRAM ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
if __name__ == "__main__":

    sites_df = utils.get_ozflux_site_list()
    for site in sites_df.index:
        site_name = site.replace(' ','')
        site_details = sites_df.loc[site]
        converter = access_data_converter(site_details)
        converter.write_to_netcdf(access_file_path)
#------------------------------------------------------------------------------