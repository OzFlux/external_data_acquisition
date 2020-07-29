#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:38:30 2018

@author: ian
"""
# System modules
from collections import OrderedDict
import copy as cp
import datetime as dt
import json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.exceptions import ConnectionError
from scipy import interpolate, signal
from time import sleep
from types import SimpleNamespace
import webbrowser
import xarray as xr

#------------------------------------------------------------------------------
### Remote configurations ###
#------------------------------------------------------------------------------

api_base_url = 'https://modis.ornl.gov/rst/api/v1/'

#------------------------------------------------------------------------------
### CLASSES ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class modis_data():

    #--------------------------------------------------------------------------
    '''
    Object containing MODIS subset data

    Args:
        * product (str): MODIS product for which to retrieve data (note that
          not all products are available from the web service - use the
          'get_product_list()' function of this module for a list of the
          available products)'
        * band (str): MODIS product band for which to retrieve data (use the
          'get_band_list(<product>)' function for a list of the available
          bands)
        * latitude (int or float): decimal latitude of location
        * longitude (int or float): decimal longitude of location
    Kwargs:
        * start_date (python datetime or None): first date for which data is
          required, or if None, first date available on server
        * end_date (python datetime): last date for which data is required,
          or if None, last date available on server
        * above_below_km (int): distance in kilometres (centred on location)
          from upper to lower boundary of subset
        * left_right_km (int): distance in kilometres (centred on location)
          from left to right boundary of subset
        * site (str): name of site to be attached to the global attributes
          of the xarray dataset
        * qcfiltered (bool): whether to eliminate observations that fail
          modis qc

    Returns:
        * MODIS data class containing the following:
            * band (attribute): MODIS band selected for retrieval
            * cellsize (attribute): actual width of pixel in m

    Todo:
        * Finish docstrings
        * Be consistent on output (dataset versus data array)
    '''

    def __init__(self, product, band, latitude, longitude,
                 start_date=None, end_date=None,
                 above_below_km=0, left_right_km=0, site=None,
                 qcfiltered=False):

        # Create a configuration namespace to pass to funcs
        config_object = get_config_obj_by_coords(product, band,
                                                 latitude, longitude,
                                                 start_date, end_date,
                                                 above_below_km, left_right_km,
                                                 site, qcfiltered)

        # Get the data
        self.data_array = request_subset_by_coords(config_object)

        # Apply range limits
        self.data_array: self._apply_range_limits()

        # QC if requested
        if qcfiltered: self._qc_data_array(config_object)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _apply_range_limits(self):

        range_limits = self.data_array.attrs['valid_range'].split('to')
        mn, mx = float(range_limits[0]), float(range_limits[-1])
        if 'scale_factor' in self.data_array.attrs:
            scale = float(self.data_array.attrs['scale_factor'])
            mn = scale * mn
            mx = scale * mx
        self.data_array = self.data_array.where((self.data_array >= mn) &
                                                (self.data_array <= mx))
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def apply_spatial_filter(self, mult=1.5):

        """Filter outliers from the 2d spatial array"""

        da = cp.deepcopy(self.data_array)
        sub_arrs = []
        for this_date in da.time:
            arr_2d = da.sel(time=this_date)
            n_valid = sum(~np.isnan(arr_2d.data)).sum()
            if n_valid == 0: next
            pct75 = np.nanpercentile(arr_2d, 75)
            pct25 = np.nanpercentile(arr_2d, 25)
            iqr = pct75 - pct25
            range_min = pct25 - mult * iqr
            range_max = pct75 + mult * iqr
            sub_arrs.append(arr_2d.where((arr_2d >= range_min) &
                                         (arr_2d <= range_max), np.nan))
        return xr.concat(sub_arrs, dim='time').transpose('y', 'x', 'time')
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def data_array_by_pixels(self, interpolate_missing=True,
                             smooth_signal=False):

        d = {}
        var_attrs = {}
        rows = self.data_array.attrs['nrows']
        cols = self.data_array.attrs['ncols']
        for i in range(rows):
            for j in range(cols):
                name_idx = str((i * rows) + j + 1)
                var_name = 'pixel_{}'.format(name_idx)
                d[var_name] = self.data_array.data[i, j, :]
                var_attrs[var_name] = {'x': self.data_array.x[j].item(),
                                       'y': self.data_array.y[i].item(),
                                       'row': i, 'col': j}
        df = pd.DataFrame(d, index=self.data_array.time.data)
        if interpolate_missing or smooth_signal: df = df.apply(_interp_missing)
        if smooth_signal: df = df.apply(_smooth_signal)
        out_xarr = df.to_xarray()
        out_xarr.attrs = self.data_array.attrs
        for var in var_attrs.keys():
            out_xarr[var].attrs = var_attrs[var]
        return out_xarr
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_pixel_by_row_col(self, row, col, filter_outliers=True,
                             interpolate_missing=True, smooth_signal=False):

        try: assert row < self.data_array.nrows
        except: raise IndexError('Row number exceeds number of rows')
        try: assert row < self.data_array.ncols
        except: raise IndexError('Column number exceeds number of columns')
        if filter_outliers: ds = self.apply_spatial_filter()
        else: ds = self.data_array
        da = self.data_array.sel(x=self.data_array.x[1], y=self.data_array.y[1])
        new_ds = _do_proc(da, locals())
        return new_ds
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_spatial_mean(self, filter_outliers=True, interpolate_missing=True,
                         smooth_signal=False):

        """Get mean time series of 2d spatial array"""

        if filter_outliers: ds = self.apply_spatial_filter()
        else: ds = self.data_array
        da = ds.mean(['x', 'y'])
        da.attrs = ds.attrs
        new_ds = _do_proc(da, locals())
        new_ds[new_ds.band].attrs['analysis'] = (
            'Mean of {0} rows x {1} columns'.format(ds.nrows, ds.ncols))
        return new_ds
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def plot_data(self, pixel='centre', plot_to_screen=True, save_to_path=None):

        pd.plotting.register_matplotlib_converters()
        state = mpl.is_interactive()
        if plot_to_screen: plt.ion()
        if not plot_to_screen: plt.ioff()
        df = self.data_array_by_pixels().to_dataframe()
        smooth_df = self.data_array_by_pixels(smooth_signal=True).to_dataframe()
        if pixel == 'centre':
            target_pixel = int(len(df.columns) / 2)
        elif isinstance(pixel, int):
            if pixel > len(df.columns) or pixel == 0:
                raise IndexError('Pixel out of range!')
            pixel = pixel - 1
            target_pixel = pixel
        else:
            raise TypeError('pixel kwarg must be either "mean" or an integer!')
        col_name = df.columns[target_pixel]
        series = df[col_name]
        smooth_series = smooth_df[col_name]
        series_label = 'centre_pixel' if pixel == 'centre' else col_name
        mean_series = df.mean(axis = 1)
        fig, ax = plt.subplots(1, 1, figsize = (14, 8))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis = 'y', labelsize = 14)
        ax.tick_params(axis = 'x', labelsize = 14)
        ax.set_xlabel('Date', fontsize = 14)
        y_label = '{0} ({1})'.format(self.data_array.attrs['band'],
                                     self.data_array.attrs['units'])
        ax.set_ylabel(y_label, fontsize = 14)
        ax.plot(df.index, df[df.columns[0]], color = 'grey', alpha = 0.1, label = 'All pixels')
        ax.plot(df.index, df[df.columns[1:]], color = 'grey', alpha = 0.1)
        ax.plot(df.index, mean_series, color = 'black', alpha = 0.5, label = 'All pixels (mean)')
        ax.plot(df.index, series, lw = 2, label = series_label)
        ax.plot(df.index, smooth_series, lw = 2,
                label = '{}_smoothed'.format(series_label))
        ax.legend(frameon = False)
        plt.ion() if state else plt.ioff()
        if save_to_path:
            try:
                fig.savefig(save_to_path)
            except FileNotFoundError:
                print('Unrecognised path!'); raise
        plt.close()
        return
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _qc_data_array(self, config_obj):

        # Function to convert qc values
        def get_bits(num):
            bin_rep = bin(int(num)).split('b')[1].zfill(8)
            bits_arr = 7 - np.array(qc_dict['bits'])[::-1]
            return int(bin_rep[bits_arr[0]: bits_arr[-1] + 1], 2)

        # Get qc variable and update config file and qc status in metadata
        qc_dict = get_qc_details(self.data_array.product)
        if not qc_dict:
            print('No QC variable defined!')
            self.data_array['qcFiltered'] = 'False'; return
        self.data_array['qcFiltered'] = 'True'

        # Check if a list was received rather than a string (handles the fact
        # that LST has separate QC variables for day and night)
        if isinstance(qc_dict['qc_name'], list):
            states = {x.split('_')[1]: x for x in qc_dict['qc_name']}
            band = config_obj.band
            qc_name = [states[x] for x in states.keys() if x in band][0]
        else:
            qc_name = qc_dict['qc_name']
        setattr(config_obj, 'band', qc_name)

        # Get qc data
        qc_array = request_subset_by_coords(config_obj)

        # Apply qc
        if not qc_dict['bits'] is None:
            vector_f = np.vectorize(get_bits)
            qc_array.data = vector_f(qc_array.data)
        max_allowed = qc_dict['reliability_threshold']
        self.data_array = self.data_array.where((qc_array <= max_allowed))
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class modis_data_network(modis_data):

    #--------------------------------------------------------------------------
    '''
    Object containing MODIS subset data

    Args:
        * product (str): MODIS product for which to retrieve data (note that
          not all products are available from the web service - use the
          'get_product_list()' function of this module for a list of the
          available products)'
        * band (str): MODIS product band for which to retrieve data (use the
          'get_band_list(<product>)' function for a list of the available
          bands)
        * network_name (str): network for which to retrieve data (use the
          'get_network_list()' function for a list of the available networks)
        * site_ID (str): network site for which to retrieve data (use the
          'get_network_list(<network>)' function for a list of the available
          sites and corresponding codes within a network)
    Kwargs:
        * start_date (python datetime or None): first date for which data is
          required, or if None, first date available on server
        * end_date (python datetime): last date for which data is required,
          or if None, last date available on server
        * qcfiltered (bool): whether or not to impose QC filtering on the data

    Returns:
        * MODIS data class containing the following:
            * band (attribute): MODIS band selected for retrieval
            * cellsize (attribute): actual width of pixel in m
    '''
    def __init__(self, product, band, network_name, site_ID,
                 start_date = None, end_date = None, qcfiltered = False):

        # Create a configuration namespace to pass to funcs
        config_object = get_config_obj_by_network_site(product, band,
                                                       network_name, site_ID,
                                                       start_date, end_date,
                                                       qcfiltered)

        # Get the data
        self.data_array = request_subset_by_siteid(config_object,
                                                   qcfiltered = qcfiltered)

        # Apply range limits
        if not qcfiltered: self._apply_range_limits()
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _apply_range_limits(self):

        range_limits = self.data_array.attrs['valid_range'].split('to')
        mn, mx = float(range_limits[0]), float(range_limits[-1])
        if 'scale_factor' in self.data_array.attrs:
            scale = float(self.data_array.attrs['scale_factor'])
            mn = scale * mn
            mx = scale * mx
        self.data_array = self.data_array.where((self.data_array >= mn) &
                                                (self.data_array <= mx))
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### FUNCTIONS ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_band_list(product, include_details = True):

    """Get available bands for a given product"""

    json_obj = requests.get(api_base_url + product + '/bands')
    band_list = json.loads(json_obj.content)['bands']
    d = OrderedDict(list(zip([x.pop('band') for x in band_list], band_list)))
    if include_details: return d
    return list(d.keys())
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_chunks(l, n = 10):

    """yield successive n-sized chunks from list l"""

    for i in range(0, len(l), n): yield l[i: i + n]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_dims_reqd_for_npixels(product, n_per_side):

    pixel_res = get_product_list(product)[product]['resolution_meters']
    return math.ceil((n_per_side - 1) * pixel_res / 1000)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_product_dates(product, lat, lng):

    """Get all available dates for given product and location"""

    req_str = "".join([api_base_url, product, "/dates?", "latitude=", str(lat),
                       "&longitude=", str(lng)])
    json_obj = requests.get(req_str)
    date_list = json.loads(json_obj.content)['dates']
    return date_list
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_product_list(include_details = True):

    """Get list of available products"""

    json_obj = requests.get(api_base_url + 'products')
    products_list = json.loads(json_obj.content)['products']
    d = OrderedDict(list(zip([x.pop('product') for x in products_list],
                        products_list)))
    if include_details: return d
    return list(d.keys())
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_product_web_page(product = None):

    """Go to web page for product"""

    products_list = get_product_list()
    modis_url_dict = {prod: '{}v006'.format(prod.lower()) for prod in
                      products_list if prod[0] == 'M'}
    viirs_url_dict = {prod: '{}v001'.format(prod.lower())
                      for prod in products_list if prod[:3] == 'VNP'}
    modis_url_dict.update(viirs_url_dict)
    base_addr = ('https://lpdaac.usgs.gov/products/{0}')
    if product is None or not product in list(modis_url_dict.keys()):
        print('Product not found... redirecting to data discovery page')
        addr = ('https://lpdaac.usgs.gov')
    else:
        addr = base_addr.format(modis_url_dict[product])
    webbrowser.open(addr)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_network_list(network = None, include_details = True):

    """Get list of available networks (if None) or sites within network if
       network name supplied"""

    if network == None:
        json_obj = requests.get(api_base_url + 'networks')
        return json.loads(json_obj.content)['networks']
    url = api_base_url + '{}/sites'.format(network)
    json_obj = requests.get(url)
    sites_list = json.loads(json_obj.content)
    d = OrderedDict(list(zip([x.pop('network_siteid') for x in sites_list['sites']],
                    sites_list['sites'])))
    if include_details: return d
    return list(d.keys())
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_pixel_subset(x_arr, pixels_per_side = 3):

    """Create a spatial subset of a larger dataset (relative to centre pixel)"""

    try:
        assert x_arr.nrows == x_arr.ncols
    except AssertionError:
        raise RuntimeError('Malformed data array!')
    if not x_arr.nrows % 2 != 0:
        raise TypeError('pixels_per_side must be an odd integer!')
    if not pixels_per_side < x_arr.nrows:
        print('Pixels requested exceeds pixels available!')
        return x_arr
    centre_pixel = int(x_arr.nrows / 2)
    pixel_min = centre_pixel - int(pixels_per_side / 2)
    pixel_max = pixel_min + pixels_per_side
    new_data = []
    for i in range(x_arr.data.shape[2]):
        new_data.append(x_arr.data[pixel_min: pixel_max, pixel_min: pixel_max, i])
    new_x = x_arr.x[pixel_min: pixel_max]
    new_y = x_arr.y[pixel_min: pixel_max]
    attrs_dict = x_arr.attrs.copy()
    attrs_dict['nrows'] = pixels_per_side
    attrs_dict['ncols'] = pixels_per_side
    attrs_dict['xllcorner'] = new_x[0].item()
    attrs_dict['yllcorner'] = new_y[0].item()
    return xr.DataArray(name = x_arr.band,
                        data = np.dstack(new_data),
                        coords = [new_y, new_x, x_arr.time],
                        dims = [ "y", "x", "time" ],
                        attrs = attrs_dict)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_qc_details(product = None):

    """Get the qc variable details for the product"""

    # Define qc dict
    d = {'M*D09A1': {'qc_name': 'sur_refl_qc_500m',
                     'bits': [0, 1], 'reliability_threshold': 1,
                     'bitmap': {'0': 'Ideal quality',
                                '1': 'Less than ideal quality',
                                '2': 'Not produced - cloud',
                                '3': 'Not produced - other'}},
         'M*D11A2': {'qc_name': ['QC_Day', 'QC_Night'],
                     'bits': [0, 1], 'reliability_threshold': 1,
                     'bitmap': {'0': 'Good data', '1': 'Marginal data',
                                '2': 'Not produced - cloud',
                                '3': 'Not produced - other'}},
         'M*D13Q1': {'qc_name': '250m_16_days_pixel_reliability',
                     'bits': None, 'reliability_threshold': 1,
                     'bitmap': {'0': 'Good data', '1': 'Marginal data',
                                '2': 'Snow/Ice', '3': 'Cloudy'}},
         'M*D15A2H': {'qc_name': 'FparLai_QC', 'bits': [5,6,7],
                      'reliability_threshold': 1,
                      'bitmap': {'0': 'Best data', '1': 'Good data, '
                                 'some saturation', '2': 'Main method failed '
                                 'due to geometry - empirical algorithm used',
                                 '3': 'Main method failed due to other - '
                                 'empirical algorithm used',
                                 '4': 'Retrieval failed'}},
         'M*D15A3H': {'qc_name': 'FparLai_QC', 'bits': [5,6,7],
                      'reliability_threshold': 1,
                      'bitmap': {'0': 'Best data', '1': 'Good data, '
                                 'some saturation', '2': 'Main method failed '
                                 'due to geometry - empirical algorithm used',
                                 '3': 'Main method failed due to other - '
                                 'empirical algorithm used',
                                 '4': 'Retrieval failed'}},
         'M*D16A2': {'qc_name': 'ET_QC_500m', 'bits': [5,6,7],
                     'reliability_threshold': 1,
                     'bitmap': {'0': 'Best data', '1': 'Good data, '
                                 'some saturation', '2': 'Main method failed '
                                 'due to geometry - empirical algorithm used',
                                 '3': 'Main method failed due to other - '
                                 'empirical algorithm used',
                                 '4': 'Retrieval failed'}},
         'M*D17A2H': {'qc_name': 'Psn_QC_500m',
                      'bits': [5,6,7], 'reliability_threshold': 4,
                      'bitmap': {'0': 'Best data', '1': 'Good data',
                                 '2': 'Substandard (geometry) - '
                                      'use with caution',
                                 '3': 'Substandard (other) - use with caution',
                                 '4': 'Not produced - non-terrestrial biome',
                                 '7': 'Fill value'}}}

    # Check product and band are legit
    if product:
        try:
            assert product in get_product_list(include_details = False)
        except AssertionError:
            print('Product not available from web service! Check available '
                  'products list using get_product_list()'); raise KeyError
    else:
        return d

    query = '{}*{}'.format(product[:1], product[2:])
    try: return d[query]
    except KeyError: return
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _do_proc(da, locals_dict):

    """Apply interpolations and smoothing to data_array, convert to dataset
       and write variable attributes """

    if locals_dict['interpolate_missing'] or locals_dict['smooth_signal']:
        da.data = _interp_missing(da.to_series())
    if locals_dict['smooth_signal']: da.data = _smooth_signal(da.to_series())
    new_ds = da.to_dataset()
    new_ds.attrs = da.attrs
    new_ds.attrs['nrows'] = 1
    new_ds.attrs['ncols'] = 1
    smoothing = ('Savitzky-Golay' if locals_dict['smooth_signal'] else 'False')
    new_ds[new_ds.band].attrs = (
        {'spatial_outliers_removed': str(locals_dict['filter_outliers']),
         'smooth_filter': smoothing,
         'interpolated': str(locals_dict['interpolate_missing'])})
    if 'x' in new_ds.coords and 'y' in new_ds.coords:
        new_ds[new_ds.band].attrs['x'] = new_ds.coords['x'].item()
        new_ds[new_ds.band].attrs['y'] = new_ds.coords['y'].item()
        new_ds = new_ds.reset_coords(['x', 'y'], drop=True)
    return new_ds
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _interp_missing(series):

    """Interpolate (Akima) signal"""

    if len(series.dropna()) < 2: return series
    days = np.array((series.index - series.index[0]).days)
    data = np.array(series)
    valid_idx = np.where(~np.isnan(data))
    if len(valid_idx[0]) == 0: return series
    f = interpolate.Akima1DInterpolator(days[valid_idx], data[valid_idx])
    return pd.Series(f(days), index=series.index)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def modis_to_from_pydatetime(date):

    """Convert between MODIS date strings and pydate format"""

    if isinstance(date, str):
        return dt.datetime.strptime(date[1:], '%Y%j').date()
    return dt.datetime.strftime(date, 'A%Y%j')
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _process_data(data, configs):

    """Process the raw data into a more human-intelligible format (xarray)"""

    # Generate metadata - note that scaling info is NOT available in the text
    # headers of the raw data when retrieved from a collection, but it can be
    # retrieved for the product (it is called 'scale_factor' instead of 'scale'
    # so here we retrieve it and rename it to scale)
    meta = {key:value for key,value in list(data[0].items())
            if key != "subset" }
    meta['product'] = configs.product
    meta['band'] = configs.band
    meta['retrieval_type'] = configs.retrieval_type
    meta['qcFiltered'] = configs.qcFiltered
    meta['site'] = configs.site
    band_attrs = get_band_list(configs.product)[configs.band]
    prod_attrs = get_product_list()[configs.product]
    prod_attrs.pop('description')
    meta.update(band_attrs)
    meta.update(prod_attrs)
    if 'scale_factor' in meta: meta['scale'] = meta.pop('scale_factor')
    if not 'scale_factor' in meta and not 'scale' in meta:
        meta['scale'] = 'Not available'

    # Iterate on data list and convert to 3D numpy array
    data_dict = {'dates': [], 'arrays': [], 'metadata': meta}
    for i in data:
        for j in i['subset']:
            if j['band'] == meta['band']:
                data_dict['dates'].append(j['calendar_date'])
                data_list = []
                for obs in j['data']:
                    try: data_list.append(float(obs))
                    except ValueError: data_list.append(np.nan)
                new_array = np.array(data_list).reshape(meta['nrows'],
                                                        meta['ncols'])
                data_dict['arrays'].append(new_array)
    stacked_array = np.dstack(data_dict['arrays'])

    # Apply scaling (if applicable - note that if data is derived from
    # collections and qc is set to True, collection has already had scaling
    # applied!)
    if not (meta['qcFiltered']=='True' and 'by_collection' in meta['retrieval_type']):
        try: stacked_array *= float(meta['scale'])
        except (TypeError, ValueError): pass

    # Assign coords and attributes
    dtdates = [dt.datetime.strptime(d,"%Y-%m-%d") for d in data_dict['dates']]
    xcoordinates = ([float(meta['xllcorner'])] +
                    [i * meta['cellsize'] + float(meta['xllcorner'])
                     for i in range(1, meta['ncols'])])
    ycoordinates = ([float(meta['yllcorner'])] +
                     [i * meta['cellsize'] + float(meta['yllcorner'])
                      for i in range(1, meta['nrows'])])
    ycoordinates = list(reversed(ycoordinates))
    return xr.DataArray(name = meta['band'], data = stacked_array,
                        coords = [np.array(ycoordinates),
                                  np.array(xcoordinates), dtdates],
                        dims = [ "y", "x", "time" ], attrs = meta)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def request_subset_by_coords(configs, qcfiltered = False):

    """Get the data from ORNL DAAC by coordinates - are we double-handling dates here?"""

    def getSubsetURL(this_start_date, this_end_date):
        return( "".join([api_base_url, configs.product, "/subset?",
                     "latitude=", str(configs.latitude),
                     "&longitude=", str(configs.longitude),
                     "&band=", configs.band,
                     "&startDate=", this_start_date,
                     "&endDate=", this_end_date,
                     "&kmAboveBelow=", str(configs.above_below_km),
                     "&kmLeftRight=", str(configs.left_right_km)]))

    dates = [x['modis_date'] for x in get_product_dates(configs.product,
                                                        configs.latitude,
                                                        configs.longitude)]
    start_idx = dates.index(configs.start_date)
    end_idx = dates.index(configs.end_date)
    date_chunks = list(_get_chunks(dates[start_idx: end_idx]))
    subsets = []
    print('Retrieving data for product {0}, band {1}:'
          .format(configs.product, configs.band))
    req_list = []
    for i, chunk in enumerate(date_chunks):
        print('[{0} / {1}] {2} - {3}'.format(str(i + 1), str(len(date_chunks)),
              chunk[0], chunk[-1]))
        url = getSubsetURL(chunk[0], chunk[-1])
        subset = request_subset_by_URLstring(url)
        subsets.append(subset)
        req_list.append(url)
    combined_array = _process_data(subsets, configs)
    combined_array.attrs['header'] = req_list
    return combined_array
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def request_subset_by_siteid(configs, qcfiltered = False):

    """Get the data from ORNL DAAC by network and site id"""

    print('Retrieving data for product {0}, band {1}'
          .format(configs.product, configs.band))
    subset_str = '/subsetFiltered?' if qcfiltered else '/subset?'
    url = (''.join([api_base_url, configs.product, '/', configs.network_name, '/',
                    configs.site_ID, subset_str, configs.band, '&startDate=',
                    configs.start_date, '&endDate=', configs.end_date]))
    subset = request_subset_by_URLstring(url)
    combined_array = _process_data([subset], configs)
    return combined_array
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def request_subset_by_URLstring(URLstr):

    """Submit request to ORNL DAAC server"""

    header = {'Accept': 'application/json'}
    for this_try in range(5):
        try:
            response = requests.get(URLstr, headers=header)
            if response.status_code == 200: return json.loads(response.text)
            if int(response.status_code / 100) == 5:
                raise ConnectionError('Server error {}'
                                      .format(str(response.status_code)))
            else:
                raise RuntimeError('Unknown error: {}'
                                   .format(str(response.status_code)))
        except ConnectionError:
            response = None
            sleep(5)
    raise ConnectionError('Connection error - server not responding after '
                          '5 tries, giving up...')
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _smooth_signal(series, n_points = 11, poly_order = 3):

    """Smooth (Savitzky-Golay) signal"""

    return signal.savgol_filter(series, n_points, poly_order, mode = "mirror")
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_config_obj_by_coords(product, band, latitude, longitude,
                             start_date=None, end_date=None,
                             above_below_km=0, left_right_km=0,
                             site=None, qcfiltered=False):

    """Create a configuration object with all required parameters for
       coordinate-based requests"""

    try:
        assert isinstance(above_below_km, int)
        assert isinstance(left_right_km, int)
    except AssertionError:
        print ('"above_below_km" and "left_right_km" kwargs must be integers')
        raise TypeError
    unique_dict = {'above_below_km': above_below_km, 'left_right_km': left_right_km,
                   'retrieval_type': 'by_coords', 'site': site}
    common_dict = _do_common_checks(product, band, latitude, longitude,
                                    start_date, end_date, qcfiltered)
    common_dict.update(unique_dict)
    return SimpleNamespace(**common_dict)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_config_obj_by_network_site(product, band, network_name, site_ID,
                                   start_date=None, end_date=None,
                                   qcfiltered=False):

    """Create a configuration object with all required parameters for
       collection-based requests"""

    # Check network and site ID are legit
    try:
        assert network_name in get_network_list()
    except AssertionError:
        print ('Network not available from web service! Check available '
               'networks list using get_network_list()'); raise KeyError
    try:
        site_attrs = get_network_list(network_name)[site_ID]
    except KeyError:
        print('Site ID code not found! Check available site ID codes '
              'using get_network_list(network)'); raise
    latitude, longitude = site_attrs['latitude'], site_attrs['longitude']
    unique_dict = {'network_name': network_name, 'site_ID': site_ID,
                   'retrieval_type': ('by_collection: {0}, {1}'
                                      .format(network_name, site_ID)),
                   'site': site_attrs['network_sitename']}
    common_dict = _do_common_checks(product, band, latitude, longitude,
                                    start_date, end_date, qcfiltered)
    common_dict.update(unique_dict)
    return SimpleNamespace(**common_dict)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _do_common_checks(*args):

    """Does basic consistency checks on the configuration arguments passed"""

    # Unpack args
    product, band = args[0], args[1]
    latitude, longitude = args[2], args[3]
    start_date, end_date = args[4], args[5]
    qcfiltered = args[6]

    # Check product and band are legit
    try:
        assert product in get_product_list(include_details = False)
    except AssertionError:
        print('Product not available from web service! Check available '
              'products list using get_product_list()'); raise KeyError
    try:
        get_band_list(product)[band]
    except KeyError:
        print('Band not available for {}! Check available bands '
              'list using get_band_list(product)'.format(product)); raise

    # Check lat and long are legit
    try:
        assert -180 <= longitude <= 180
        assert -90 <= latitude <= 90
    except AssertionError:
        print ('Latitude or longitude out of bounds'); raise RuntimeError

    # Check qcfiltered kwarg is bool
    try: assert isinstance(qcfiltered, bool)
    except AssertionError():
        print ('"qcfiltered" kwarg must be of type bool'); raise TypeError

    # Check and set MODIS dates
    avail_dates = get_product_dates(product, latitude, longitude)
    py_avail_dates = np.array([dt.datetime.strptime(x['modis_date'], 'A%Y%j').date()
                               for x in avail_dates])
    if start_date:
        py_start_dt = dt.datetime.strptime(start_date, '%Y%m%d').date()
    else:
        py_start_dt = py_avail_dates[0]
    if end_date:
        py_end_dt = dt.datetime.strptime(end_date, '%Y%m%d').date()
    else:
        py_end_dt = py_avail_dates[-1]
    start_idx = abs(py_avail_dates - py_start_dt).argmin()
    end_idx = abs(py_avail_dates - py_end_dt).argmin()
    return {'product': product, 'band': band,
            'latitude': latitude, 'longitude': longitude,
            'start_date': avail_dates[start_idx]['modis_date'],
            'end_date': avail_dates[end_idx]['modis_date'],
            'qcFiltered': 'True' if qcfiltered else False}
#------------------------------------------------------------------------------