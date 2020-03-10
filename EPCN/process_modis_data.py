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
import os
import pandas as pd
import sys

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
def _product_band_to_retrieve():

    return {'MOD09A1': ['sur_refl_b07'],
            'MOD11A2': ['LST_Day_1km', 'LST_Night_1km'],
            'MOD13Q1': ['250m_16_days_EVI', '250m_16_days_NDVI'],
            'MCD15A3H': ['Lai_500m', 'Fpar_500m'],
            'MOD16A2': ['ET_500m'],
            'MOD17A2H': ['Gpp_500m']}
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
def _set_var_attrs():
    
    {'sg_order': '5',
     'evi_interpolate': 'linear',
     'sg_num_points': '5',
     'instrument': 'not defined',
     'valid_range': '-1e+35,1e+35',
     'ancillary_variables': 'not defined',
     'evi_sd_threshold': '0.05',
     'horiz_resolution': '250m',
     'height': 'not defined',
     'long_name': 'MODIS EVI, smoothed and interpolated',
     'standard_name': 'not defined',
     'evi_quality_threshold': '1',
     'units': 'none',
     'serial_number': 'not defined',
     'cutout_size': '3',
     'evi_smooth_filter': 'Savitsky-Golay'}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
if __name__ == "__main__":

    # Get sites info for processing
    sites = utils.get_ozflux_site_list(master_file_path)

    # Get list of ozflux sites that are in the MODIS collection (note Wombat
    # has designated site name 'Wombat', so change in dict)
    ozflux_modis_collection_sites = mfr.get_network_list('OZFLUX')
    coll_dict = {ozflux_modis_collection_sites[x]['network_sitename']:
                 x for x in ozflux_modis_collection_sites.keys()}
    coll_dict['Wombat State Forest'] = coll_dict.pop('Wombat')

    # Iterate on product (create dirs where required)
    products_dict = _product_band_to_retrieve()
    for product in products_dict:
        this_path = os.path.join(output_path, product)
        if not os.path.exists(this_path): os.makedirs(this_path)

        # Iterate on band
        for band in products_dict[product]:
            short_name = _band_short_name(band)

            # Get site data and write to netcdf
            for site in sites.index:
                site_details = sites.loc[site]
                print('Retrieving data for site {}:'.format(site))
                target = os.path.join(this_path,
                                      '{0}_{1}'.format(site.replace(' ', ''),
                                                       short_name))
                full_nc_path = target + '.nc'
                full_plot_path = target + '.png'
                try:
                    first_date = dt.date(int(sites.loc[site, 'Start year']) - 1, 7, 1)
                    first_date_modis = dt.datetime.strftime(first_date, '%Y%m%d')
                except (TypeError, ValueError): first_date_modis = None
                try:
                    last_date = dt.date(int(sites.loc[site, 'End year']) + 1, 6, 1)
                    last_date_modis = dt.datetime.strftime(last_date, '%Y%m%d')
                except (TypeError, ValueError): last_date_modis = None

                # Get sites in the collection
                if site in coll_dict.keys():
                    site_code = coll_dict[site]
                    x = mfr.modis_data_network(product, band, 'OZFLUX', site_code,
                                               first_date_modis, last_date_modis,
                                               qcfiltered=True)

                # Get sites not in the collection
                else:
                    km_dims = mfr.get_dims_reqd_for_npixels(product, 5)
                    x = mfr.modis_data(product, band, site_details.Latitude, 
                                       site_details.Longitude, first_date_modis, 
                                       last_date_modis, km_dims, km_dims, site, 
                                       qcfiltered=True)

                # Reduce the number of pixels to 3 x 3
                x.data_array = mfr.get_pixel_subset(x.data_array,
                                                    pixels_per_side = 3)

                # Get outputs and write to file (plots then nc)
                x.plot_data(plot_to_screen=False, save_to_path=full_plot_path)
                ds = (pd.DataFrame({short_name: x.get_spatial_mean(),
                                    short_name + '_smoothed': x.get_spatial_mean(smooth_signal=True)})
                      .to_xarray())
                ds.attrs = x.data_array.attrs
                str_step = str(int(site_details['Time step'])) + 'T'
                resampled_ds = ds.resample({'time': str_step}).interpolate()
                resampled_ds.time.encoding = {'units': 'days since 1800-01-01',
                                              '_FillValue': None}
                _set_global_attrs(resampled_ds, site_details)
                resampled_ds.to_netcdf(full_nc_path, format='NETCDF4')
