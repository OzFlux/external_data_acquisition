#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:41:50 2019

@author: ian
"""
import configparser
import logging
import os
import pandas as pd
from timezonefinder import TimezoneFinder as tzf
import xlrd

#------------------------------------------------------------------------------
def get_configs():

    """Create a configuration file that defines requisite read and write paths
       (reads from same dir as executing script - i.e. this one!)"""
    
    path = os.path.join(os.path.dirname(__file__), 'paths.ini')
    config = configparser.ConfigParser()
    config.read(path)
    return config
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_ozflux_site_list(master_file_path = None):

    """Generates a dataframe containing requisite site details (auto-reads 
       file location from [DEFAULT][site_details] in paths.ini - see 
       get_configs)"""
    
    if not master_file_path:
        configs = get_configs()
        master_file_path = configs['DEFAULT']['site_details']
    wb = xlrd.open_workbook(master_file_path)
    sheet = wb.sheet_by_name('Active')
    header_row = 9
    header_list = sheet.row_values(header_row)
    df = pd.DataFrame()
    for var in ['Site', 'Latitude', 'Longitude', 'Altitude', 'Time step', 
                'Start year', 'End year']:
        index_val = header_list.index(var)
        df[var] = sheet.col_values(index_val, header_row + 1)
    df['Start year'] = pd.to_numeric(df['Start year'], errors='coerce')
    df['End year'] = pd.to_numeric(df['End year'], errors='coerce')
    df.index = df[header_list[0]]
    df.drop(header_list[0], axis = 1, inplace = True)
    df['Time zone'] = [tzf().timezone_at(lng=x[0], lat=x[1]) 
                       for x in zip(df.Longitude, df.Latitude)]
    return df
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
def set_logger(target_filepath):

    """Does basic standardised configuration for logger"""
    
    logging.basicConfig(filename=target_filepath,
                        format='%(levelname)s %(message)s',
                        level=logging.DEBUG)
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s %(message)s')
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
#------------------------------------------------------------------------------