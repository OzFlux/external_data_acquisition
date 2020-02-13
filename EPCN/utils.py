#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:41:50 2019

@author: ian
"""
import configparser
from datetime import datetime
import logging
import os
import pandas as pd
import time
import xlrd

#------------------------------------------------------------------------------
def get_configs():

    path = os.path.join(os.path.dirname(__file__), 'paths.ini')
    config = configparser.ConfigParser()
    config.read(path)
    return config
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_ozflux_site_list(master_file_path = None):

    if not master_file_path:
        configs = get_configs()
        master_file_path = configs['DEFAULT']['site_details']
    wb = xlrd.open_workbook(master_file_path)
    sheet = wb.sheet_by_name('Active')
    header_row = 9
    header_list = sheet.row_values(header_row)
    df = pd.DataFrame()
    for var in ['Site', 'Latitude', 'Longitude', 'Time step', 'Start year',
                'End year']:
        index_val = header_list.index(var)
        df[var] = sheet.col_values(index_val, header_row + 1)
    df['Start year'] = pd.to_numeric(df['Start year'], errors='coerce')
    df['End year'] = pd.to_numeric(df['End year'], errors='coerce')
    df.index = df[header_list[0]]
    df.drop(header_list[0], axis = 1, inplace = True)
    return df
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
def set_logger(target_filepath):

    t = time.localtime()
    rundatetime = (datetime(t[0],t[1],t[2],t[3],t[4],t[5]).strftime("%Y%m%d%H%M"))
    log_filename = os.path.join(target_filepath, 
                                'access_data_{}.log'.format(rundatetime))
    logging.basicConfig(filename=log_filename,
                        format='%(levelname)s %(message)s',
                        level=logging.DEBUG)
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s %(message)s')
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
#------------------------------------------------------------------------------