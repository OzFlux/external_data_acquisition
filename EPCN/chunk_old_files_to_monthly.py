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

import utils

def get_date_indices(date_idx, time_zone, time_step):
    
    add_time = timezone(time_zone).fromutc(dt.datetime(2015, 6, 1))
    start_increment = add_time.time()
    end_increment = (add_time - dt.timedelta(minutes=time_step)).time()
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
                          [dt.datetime.combine(dt.date(*x), end_increment) 
                           for x in end_list]))
    for dates in dates_list: yield dates
        
read_path = '/rdsi/market/access_old_site_files'
write_path = '/rdsi/market/access_old_site_files/monthly'

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
    date_idx = get_date_indices(ds.time, sd['Time zone'], sd['Time step'])
    for dates in date_idx:
        print ('Processing date {}'.format(dates[0]))
        sub_ds = ds.sel(time=slice(dates[0], dates[1]))
        if len(sub_ds.time) == 0: continue
        ym_str = dt.datetime.strftime(dates[0], '%Y%m')
        target_dir = os.path.join(write_path, ym_str)
        if not os.path.isdir(target_dir): os.mkdir(target_dir)
        this_write_path = os.path.join(target_dir, 
                                       '{}_ACCESS_{}.nc'.format(strp_site,
                                                                ym_str))
        start_date = dt.datetime.strftime(dates[0], '%Y-%m-%d %H:%M:%S')
        end_date = dt.datetime.strftime(dates[-1], '%Y-%m-%d %H:%M:%S')
        rec_len = len(sub_ds.time)
        sub_ds['start_date'] = start_date
        sub_ds['end_date'] = end_date
        sub_ds['nc_nrecs'] = rec_len
        sub_ds.to_netcdf(this_write_path, format='NETCDF4')