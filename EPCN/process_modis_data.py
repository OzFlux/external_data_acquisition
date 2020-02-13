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

sys.path.append('../MODIS')
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
if __name__ == "__main__":

    # Get sites info for processing
    sites = utils.get_ozflux_site_list(master_file_path)

    products_dict = _product_band_to_retrieve()

    # Get list of ozflux sites that are in the MODIS collection (note Wombat
    # has designated site name 'Wombat', so change in dict)
    ozflux_modis_collection_sites = mfr.get_network_list('OZFLUX')
    coll_dict = {ozflux_modis_collection_sites[x]['network_sitename']:
                 x for x in ozflux_modis_collection_sites.keys()}
    coll_dict['Wombat State Forest'] = coll_dict.pop('Wombat')

    # Iterate on product (create dirs where required)
    for product in products_dict:
        this_path = os.path.join(output_path, product)
        if not os.path.exists(this_path): os.makedirs(this_path)

        # Iterate on band
        for band in products_dict[product]:

            short_name = _band_short_name(band)

            # Get site data and write to netcdf
            for site in sites.index:

                print('Retrieving data for site {}:'.format(site))

                target = os.path.join(this_path,
                                      '{0}_{1}'.format(site.replace(' ', '_'),
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
                    x = mfr.modis_data(product, band,
                                       sites.loc[site, 'Latitude'],
                                       sites.loc[site, 'Longitude'],
                                       first_date_modis, last_date_modis,
                                       km_dims, km_dims, site, qcfiltered=True)

                # Reduce the number of pixels to 5 x 5
                x.data_array = mfr.get_pixel_subset(x.data_array,
                                                    pixels_per_side = 5)

                # Get outputs and write to file (plots then nc)
                x.plot_data(plot_to_screen=False, save_to_path=full_plot_path)
                da = (pd.DataFrame({short_name: x.get_spatial_mean(),
                                    short_name + '_smoothed': x.get_spatial_mean(smooth_signal=True)})
                      .to_xarray())
                da.attrs = x.data_array.attrs
                resampled_da = da.resample({'time': '30T'}).interpolate()
                resampled_da.time.encoding = {'units': 'days since 1800-01-01',
                                              '_FillValue': None}
                resampled_da.to_netcdf(full_nc_path, format='NETCDF4')