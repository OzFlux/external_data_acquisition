#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:32:50 2020

@author: imchugh
"""

#------------------------------------------------------------------------------
### MODULES (STANDARD) ###
#------------------------------------------------------------------------------

import datetime as dt
import numpy as np
import os
import pandas as pd
from requests.exceptions import ConnectionError
import sys
import xarray as xr

#------------------------------------------------------------------------------
### MODULES (CUSTOM) ###
#------------------------------------------------------------------------------

this_path = os.path.join(os.path.dirname(__file__), '../MODIS')
sys.path.append(this_path)
import modis_functions_rest as mfr
import utils

#------------------------------------------------------------------------------
### CONFIGURATIONS ###
#------------------------------------------------------------------------------

configs = utils.get_configs()
master_file_path = configs['DEFAULT']['site_details']
output_path = configs['nc_data_write_paths']['modis']

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### MAIN PROGRAM
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_alias_dict():

    """Return dict of key: value pairs where key is name used by modis, value
       is name used by OzFlux"""

    return {'Wombat': 'Wombat State Forest',
            'Alpine Peatland': 'Alpine Peat',
            'Arcturus Emerald': 'Arcturus Emerald'}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_site_details(site_details, product, band_short_name):

    use_name = site_details.name.replace(' ', '')
    dir_path = os.path.join(output_path.format(use_name), product)
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    files_path = os.path.join(dir_path, 
                              '{0}_{1}'.format(use_name, band_short_name))
    site_details['full_nc_path'] = files_path + '.nc'
    site_details['full_plot_path'] = files_path + '.png'
    try:
        first_date = dt.date(int(sites.loc[site, 'Start year']) - 1, 7, 1)
        first_date_modis = dt.datetime.strftime(first_date, '%Y%m%d')
    except (TypeError, ValueError): first_date_modis = None
    site_details['first_date_modis'] = first_date_modis
    try:
        last_date = dt.date(int(sites.loc[site, 'End year']) + 1, 6, 1)
        last_date_modis = dt.datetime.strftime(last_date, '%Y%m%d')
    except (TypeError, ValueError): last_date_modis = None
    site_details['last_date_modis'] = last_date_modis
    return site_details
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _band_short_name(band):

    d = {'sur_refl_b07': 'reflectance_b7', 'LST_Day_1km': 'LST_Day',
         'LST_Night_1km': 'LST_night', '250m_16_days_EVI': 'EVI',
         '250m_16_days_NDVI': 'NDVI', 'Lai_500m': 'LAI', 'Fpar_500m': 'FPAR',
         'ET_500m': 'ET', 'Gpp_500m': 'GPP'}
    return d[band]
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
def _product_band_to_retrieve():

    return {'MOD09A1': ['sur_refl_b07'],
            'MOD11A2': ['LST_Day_1km', 'LST_Night_1km'],
            'MOD13Q1': ['250m_16_days_EVI', '250m_16_days_NDVI'],
            'MCD15A3H': ['Lai_500m', 'Fpar_500m'],
            'MOD16A2': ['ET_500m'],
            'MOD17A2H': ['Gpp_500m']}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _set_global_attrs(ds, site_details):

    start_date = dt.datetime.strftime(pd.to_datetime(ds.time[0].item()),
                                      '%Y-%m-%d %H:%M:%S')
    end_date = dt.datetime.strftime(pd.to_datetime(ds.time[-1].item()),
                                    '%Y-%m-%d %H:%M:%S')
    run_date = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S')
    nc_nrecs = len(ds.time)

    d = {'start_date': start_date,
         'end_date': end_date,
         'nc_nrecs': nc_nrecs,
         'site_name': ds.attrs.pop('site').replace(' ',''),
         'time_step': str(int(site_details['Time step'])),
         'time_zone': site_details['Time zone'],
         'nc_rundatetime': run_date}

    ds.attrs.update(d)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _set_var_attrs(ds):

    for this_var in list(ds.variables):
        if this_var in ds.dims: continue
        ds[this_var].attrs['units'] = ds.units
        ds[this_var].attrs['valid_range'] = '1e+35,-1e+35'
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
if __name__ == "__main__":

    # Get sites info for processing
    sites = utils.get_ozflux_site_list(active_sites_only=False)

    # Get list of ozflux sites that are in the MODIS collection (note Wombat
    # has designated site name 'Wombat', Alpine Peat has designated site name
    # Alpine Peatland, so change both in dict)
    ozflux_modis_collection_sites = mfr.get_network_list('OZFLUX')
    coll_dict = {ozflux_modis_collection_sites[x]['network_sitename']:
                 x for x in ozflux_modis_collection_sites.keys()}
    alias_dict = _get_alias_dict()
    for key in alias_dict: coll_dict[alias_dict[key]] = coll_dict.pop(key)

    # Iterate on product (create dirs where required)
    products_dict = _product_band_to_retrieve()
    for product in products_dict:

        # Iterate on band
        for band in products_dict[product]:
            short_name = _band_short_name(band)

        # Get site data and write to netcdf
            for site in sites.index:
                print('Retrieving data for site {}:'.format(site))
                site_details = _get_site_details(sites.loc[site].copy(), 
                                                 product, band)

                # Try to parse the data; catch server errors and continue
                try:
                    # Get sites in the collection
                    try:
                        site_code = coll_dict[site]
                        x = mfr.modis_data_network(
                            product, band, 'OZFLUX', site_code,
                            site_details['first_date_modis'],
                            site_details['last_date_modis'], qcfiltered=True
                            )

                # Get sites not in the collection
                    except KeyError:
                        km_dims = mfr.get_dims_reqd_for_npixels(product, 5)
                        x = mfr.modis_data(
                            product, band, site_details.Latitude,
                            site_details.Longitude,
                            site_details['first_date_modis'],
                            site_details['last_date_modis'],
                            km_dims, km_dims, site, qcfiltered=True
                            )
                except ConnectionError as e:
                    print ('Data retrieval failed for site {0} with error'
                           '{1}'.format(site, e))
                    continue

                # Reduce the number of pixels to 3 x 3
                x.data_array = mfr.get_pixel_subset(x.data_array,
                                                    pixels_per_side = 3)

                # Get outputs and write to file (plots then nc)
                x.plot_data(plot_to_screen=False,
                            save_to_path=site_details['full_plot_path'])
                ds = xr.merge([x.get_spatial_mean().rename({band: short_name}),
                               x.get_spatial_mean(smooth_signal=True)
                               .rename({band: short_name + '_smoothed'})])
                ds.attrs = x.data_array.attrs
                _set_var_attrs(ds)
                str_step = str(int(site_details['Time step'])) + 'T'
                resampled_ds = ds.resample({'time': str_step}).interpolate()
                resampled_ds.time.encoding = {'units': 'days since 1800-01-01',
                                              '_FillValue': None}
                final_ds = xr.merge([resampled_ds, make_qc_flags(resampled_ds)])
                final_ds.attrs = resampled_ds.attrs
                _set_global_attrs(final_ds, site_details)
                final_ds.to_netcdf(site_details['full_nc_path'], format='NETCDF4')