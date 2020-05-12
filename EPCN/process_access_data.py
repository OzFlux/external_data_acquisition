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
import met_funcs

#------------------------------------------------------------------------------
### CONFIGURATIONS ###
#------------------------------------------------------------------------------

configs = utils.get_configs()
raw_file_path = configs['raw_data_write_paths']['access']
raw_file_path_prev = configs['raw_data_write_paths']['access_previous']
nc_write_path = configs['nc_data_write_paths']['access']

#------------------------------------------------------------------------------
### CLASSES ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class access_data_converter():

    def __init__(self, site_details, include_prior_data=False):

        self.site_details = site_details
        self.include_prior_data = include_prior_data

    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def create_dataset(self):

        # Do formatting, conversions and attributes
        ds = self.get_raw_file()
        ds = ds.compute() # This converts from dask to numpy :)
        ds = ds[list(vars_dict.keys())]
        _apply_range_limits(ds)
        ds = ds.rename(vars_dict)
        ds = _reindex_time(ds)
        if self.site_details['Time step'] == 30: ds = _resample_dataset(ds)
        do_conversions(ds)
        get_energy_components(ds)
        ds = ds.fillna(-9999.0)
        offset = self.get_utc_offset()
        ds.time.data = (pd.to_datetime(ds.time.data) + dt.timedelta(hours=offset))
        _set_var_attrs(ds)

        # Rebuild dataset with separated pixels
        ds_list = []
        for i, this_lat in enumerate(ds.lat):
            for j, this_lon in enumerate(ds.lon):

                # Deal with dimensions
                sub_ds = ds.sel(lat=ds.lat[i], lon=ds.lon[j])
                for x in ['Ts', 'Sws']:
                    sub_ds[x] = sub_ds[x].sel(soil_lvl=sub_ds.soil_lvl[0])

                # Dump extraneous dims and coords
                sub_ds = sub_ds.drop_dims(['soil_lvl'])
                sub_ds = sub_ds.reset_coords(['lat', 'lon'], drop=True)

                # Add lat long to variable attributes
                for var in sub_ds:
                    sub_ds[var].attrs['latitude'] = round(this_lat.item(), 4)
                    sub_ds[var].attrs['longitude'] = round(this_lon.item(), 4)

                # Rename with variable numbering
                var_suffix = '_{}{}'.format(str(i), str(j))
                new_dict = {x: x + var_suffix for x in list(sub_ds.variables)
                            if not x == 'time'}
                sub_ds = sub_ds.rename(new_dict)

                # Append
                ds_list.append(sub_ds)

        # Merge qc flags
        final_ds = xr.merge(ds_list)
        final_ds = xr.merge([final_ds, make_qc_flags(final_ds)])
        
        # Add old data (or not), set global attrs and return
        if self.include_prior_data:
            try: final_ds = _combine_datasets(final_ds, self.site_details.name)
            except OSError: pass
        _set_global_attrs(final_ds, self.site_details)
        return final_ds

    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_file_list(self):

        search_str = self.site_details.name.replace(' ', '')
        return sorted(glob.glob(raw_file_path +
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

        tz_obj = timezone(self.site_details['Time zone'])
        now_time = dt.datetime.now()
        return (tz_obj.utcoffset(now_time) - tz_obj.dst(now_time)).seconds / 3600
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def write_to_netcdf(self, write_path):

        print('Writing netCDF file for site {}'.format(self.site_details.name))
        dataset = self.create_dataset()
        fname = '{}_ACCESS.nc'.format(self.site_details.name.replace(' ', ''))
        target = os.path.join(write_path, fname)
        dataset.to_netcdf(target, format='NETCDF4')
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### FUNCTIONS ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _apply_range_limits(ds):

    for var in ds.variables:
        if var in ds.dims: continue
        lims = range_dict[var]
        ds[var] = ds[var].where(cond=(ds[var] >= lims[0]) & (ds[var] <= lims[1]))
    ds['inst_prcp'] = ds.inst_prcp.where(cond=((ds.inst_prcp < -1) |
                                               (ds.inst_prcp > 0.001)), other=0)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _collate_prior_data(site_str):
    
    """Stack the individual data files from the old ACCESS collection together;
       note that we suspect the rainfall calculation for the mod6 = 0 hours is
       wrong in this data, so we cut to the new collection in November of '19,
       the first month for which we collected complete data"""
    
    def preproc(sub_ds):
        return sub_ds.drop_sel(time=sub_ds.time[-1].data)

    cutoff_month = '201910'
    f_list = sorted(glob.glob(raw_file_path_prev + '/**/{}*'.format(site_str)))
    if len(f_list) == 0: raise OSError
    months = [os.path.splitext(x)[0].split('_')[-1] for x in f_list]
    for i, month in enumerate(months): 
        if month > cutoff_month: break
    f_list = f_list[:i]
    prior_ds = xr.open_mfdataset(f_list, combine='by_coords', preprocess=preproc).compute()
    for var in list(prior_ds.variables):
        if not 'Precip' in var: continue
        prior_ds[var] = prior_ds[var].where(cond=prior_ds[var]<9999, other=-9999)
    return prior_ds
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _combine_datasets(current_ds, site_name):
    
    site_name = site_name.replace(' ','')
    prior_ds = _collate_prior_data(site_name)
    prior_ds = prior_ds.drop(labels=[x for x in prior_ds.variables 
                                     if not x in current_ds.variables])
    for var in current_ds.variables:
        if var in current_ds.dims: continue
        current_ds[var].attrs = prior_ds[var].attrs
    current_ds = xr.concat([prior_ds, current_ds], dim='time')
    idx = np.unique(current_ds.time.data, return_index=True)[1]
    current_ds = current_ds.isel(time=idx)
    return current_ds
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
def do_conversions(ds):

    ds['Ws'] = met_funcs.get_ws_from_uv(ds.u, ds.v)
    ds['Wd'] = met_funcs.get_wd_from_uv(ds.u, ds.v)
    ds['ps'] = met_funcs.convert_Pa_to_kPa(ds.ps)
    ds['RH'] = (met_funcs.get_e_from_q(ds.q, ds.ps) / 
                met_funcs.get_es(ds.Ta)) * 100
    ds['Ah'] = met_funcs.get_Ah(ds.Ta, ds.q, ds.ps)
    ds['Ta'] = met_funcs.convert_Kelvin_to_celsius(ds.Ta)
    ds['Ts'] = met_funcs.convert_Kelvin_to_celsius(ds.Ts) 
    ds['Sws'] = ds['Sws'] / 100
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_energy_components(ds):

    ds['Fsu'] = ds.Fsd - ds.Fn_sw
    ds['Flu'] = ds.Fld - ds.Fn_lw
    ds['Fn'] = (ds.Fsd - ds.Fsu) + (ds.Fld - ds.Flu)
    ds['Fa'] = ds.Fh + ds.Fe
    ds['Fg'] = ds.Fn - ds.Fa
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
def _reindex_time(ds):

    new_index = pd.date_range(ds.time[0].item(), ds.time[-1].item(), freq='60T')
    return ds.reindex(time=new_index)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _resample_dataset(ds):

    """Resample to half-hourly and interpolate only gaps created by
       resampling; note that rainfall must be converted to a cumulative sum
       to be resampled, then redifferenced to recover hourly total rainfall"""

    new_ds = ds.copy()
    new_ds['cml_precip'] = new_ds.Precip.cumsum(dim='time')
    new_dates = pd.date_range(start=ds.time[0].item(), end=ds.time[-1].item(),
                              freq='30T')
    new_ds = (new_ds.reindex(time=new_dates)
              .interpolate_na(dim='time', max_gap=pd.Timedelta(hours=1)))
    new_ds['cml_precip'] = new_ds.cml_precip.where(~np.isnan(new_ds.Precip))
    new_ds['Precip'] = new_ds.cml_precip - new_ds.cml_precip.shift(time=1)
    new_ds = new_ds.drop('cml_precip')
    return new_ds
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _set_var_attrs(ds):

    generic_dict = {'instrument': '', 'valid_range': (-1e+35,1e+35),
                    'missing_value': -9999, 'height': '',
                    'standard_name': '', 'group_name': '',
                    'serial_number': ''}

    for this_var in list(ds.variables):
        if this_var in ds.dims: continue
        base_dict = generic_dict.copy()
        try:
            base_dict.update(attrs_dict[this_var])
        except KeyError:
            base_dict.update(attrs_dict[this_var.split('_')[0]])
        ds[this_var].attrs = base_dict
        ds[this_var].encoding = {'_FillValue': -9999}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _set_global_attrs(ds, site_details):

    ds.attrs = {'nc_nrecs': len(ds.time),
                'start_date': (dt.datetime.strftime
                               (pd.Timestamp(ds.time[0].item()),
                                '%Y-%m-%d %H:%M:%S')),
                'end_date': (dt.datetime.strftime
                             (pd.Timestamp(ds.time[-1].item()),
                              '%Y-%m-%d %H:%M:%S')),
                'latitude': site_details.Latitude,
                'longitude': site_details.Longitude,
                'site_name': site_details.name,
                'time_step': site_details['Time step'],
                'xl_datemode': '0'}
    ds.time.encoding = {'units': 'days since 1800-01-01',
                        '_FillValue': None}
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
              'inst_prcp': [-1, 100],
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

    sites = utils.get_ozflux_site_list()
    for site in sites.index:
        specific_file_path = nc_write_path.format(site.replace(' ', ''))
        converter = access_data_converter(sites.loc[site], include_prior_data=True)
        converter.write_to_netcdf(specific_file_path)
#------------------------------------------------------------------------------