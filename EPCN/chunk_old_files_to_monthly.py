#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:04:40 2020

@author: imchugh
"""

import datetime as dt
import glob
import numpy as np
import os
from pytz import timezone
import xarray as xr
import pdb

import utils

def get_date_indices(date_idx, time_zone):
    
    add_time = timezone(time_zone).fromutc(dt.datetime(2015, 6, 1))
    start_increment = add_time.time()
    start_date = date_idx[0].values.astype('datetime64[s]').tolist()
    end_date = date_idx[-1].values.astype('datetime64[s]').tolist()
    start_list = []
    for this_year in range(start_date.year, end_date.year + 1):
        start_list += (
                list(zip(np.tile(this_year, 12), np.arange(1, 13), 
                         np.tile(1, 12)))
            )
    end_list = start_list[1:]
    end_list.append((this_year + 1, 1, 1))
    dates_list = list(zip([dt.datetime.combine(dt.date(*x), start_increment) 
                           for x in start_list], 
                          [dt.datetime.combine(dt.date(*x), start_increment) 
                           for x in end_list]))
    for dates in dates_list: yield dates
        
read_path = '/rdsi/market/access_old_site_files'
write_path = '/rdsi/market/access_old_site_files/monthly'
ref_file_path = '/rdsi/market/access_opendap/monthly/201505'

sites = utils.get_ozflux_site_list()
for site in sites.index:
    sd = sites.loc[site]
    print ('Getting data for site {}'.format(site))
    strp_site = site.replace(' ','')
    f_list = glob.glob(read_path + '/' + strp_site + '*split*')
    if len(f_list) == 0: continue
    f_name = f_list[0]
    ds = xr.open_dataset(f_name)
    if len(ds.time) == 0: continue
    try:
        ref_ds = xr.open_dataset(os.path.join(ref_file_path, 
                                              '{}_ACCESS_201505.nc'.format(strp_site)))
    except FileNotFoundError:
        continue
    attrs = ref_ds.attrs
    the_vars = list(ref_ds.variables)
    ref_ds.close()
    date_idx = get_date_indices(ds.time, sd['Time zone'])
    for dates in date_idx:
        print ('Processing date {}'.format(dates[0]))
        sub_ds = ds.sel(time=slice(dates[0], dates[1]))
        if len(sub_ds.time) == 0: continue
        ym_str = dt.datetime.strftime(dates[0], '%Y%m')
        if int(ym_str) >= 201505: break
        target_dir = os.path.join(write_path, ym_str)
        if not os.path.isdir(target_dir): os.mkdir(target_dir)
        this_write_path = os.path.join(target_dir, 
                                       '{}_ACCESS_{}.nc'.format(strp_site,
                                                                ym_str))
        anotinb = [x for x in list(sub_ds.variables) if not x in the_vars]
        sub_ds = sub_ds.drop(anotinb)
        if len(sub_ds.dims) == 3:
            sub_ds = sub_ds.sel(latitude=sub_ds.latitude[0], 
                                longitude=sub_ds.longitude[0])
        rec_len = len(sub_ds.time)
        sub_ds.attrs = attrs
        sub_ds.attrs['nc_nrecs'] = rec_len
        sub_ds.to_netcdf(this_write_path, format='NETCDF4')