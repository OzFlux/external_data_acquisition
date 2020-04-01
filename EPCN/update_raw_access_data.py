#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:19:53 2019

@author: ian
"""

#------------------------------------------------------------------------------
### MODULES (STANDARD) ###
#------------------------------------------------------------------------------

from bs4 import BeautifulSoup
import datetime as dt
import logging
import netCDF4
import numpy as np
import os
import pandas as pd
import requests
from subprocess import call as spc
import sys

#------------------------------------------------------------------------------
### MODULES (CUSTOM) ###
#------------------------------------------------------------------------------

this_path = os.path.join(os.path.dirname(__file__), '../EPCN')
sys.path.append(this_path)
import utils

#------------------------------------------------------------------------------
### CONFIGURATIONS (LOCAL) ###
#------------------------------------------------------------------------------

configs = utils.get_configs()
base_dir = configs['nc_data_write_paths']['access']
master_file_path = configs['DEFAULT']['site_details']
base_log_path = configs['DEFAULT']['log_path']

#------------------------------------------------------------------------------
### CONFIGURATIONS (REMOTE) ###
#------------------------------------------------------------------------------

retrieval_path = 'http://opendap.bom.gov.au:8080/thredds/{}/bmrc/access-r-fc/ops/surface/'

#------------------------------------------------------------------------------
### FUNCTIONS ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def check_seen_files(opendap_url, base_dir, site_list):

    """Check local files to see which of the available files on the opendap
       server have been seen and processed"""

    site_list = [x.replace(' ','') for x in site_list]
    opendap_files = _list_opendap_dirs(opendap_url)
    month_dirs = np.unique([x[:6] for x in opendap_files]).astype(str)
    check_paths = [os.path.join(base_dir, 'Monthly_files', x) for x in month_dirs]
    data = (np.tile(False, len(opendap_files) * len(site_list))
            .reshape(len(opendap_files), len(site_list)))
    seen_df = pd.DataFrame(data, index = opendap_files, columns = site_list)
    for site in seen_df.columns:
        seen_dirs = []
        for target_path in check_paths:
            target = os.path.join(target_path, '{}.nc'.format(site))
            try:
                nc = netCDF4.Dataset(target)
                dts = sorted(netCDF4.num2date(nc.variables['time'][:],
                             units = nc.variables['time'].units))
                seen_dates = [dt.datetime.strftime(x, '%Y%m%d') for x in dts]
                seen_hours = [str(x.hour - x.hour % 6).zfill(2) for x in dts]
                seen_dirs += list(set([x[0] + x[1] for x in zip(seen_dates,
                                                                seen_hours)]))
                nc.close()
            except IOError:
                continue
        seen_df[site] = list(map(lambda x: x in seen_dirs, opendap_files))
    seen_df = seen_df.T
    seen_dict = {}
    for site in seen_df.columns:
        l = list(seen_df[seen_df[site]==False].index)
        if l:
            seen_dict[site] = l
    return seen_dict
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def check_set_subdirs(base_dir):

    """Check if required directories reside in base directory
       and create if not"""

    logging.info('Checking for requisite directories')
    missing_dirs = []
    for sub_dir in ['Continental_files', 'Monthly_files',
                    'Precip_forecast_files', 'Working_files', 'Log_files']:
        expected_path = os.path.join(base_dir, sub_dir)
        if os.path.exists(expected_path): continue
        os.makedirs(expected_path)
        missing_dirs.append(sub_dir)
    if not missing_dirs:
        logging.info('All required subdirectories present')
    else:
        missing_str = ', '.join(missing_dirs)
        logging.info('The following directories were missing and have been '
                     'created: {}'.format(missing_str))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _list_opendap_dirs(url):

    """Scrape list of directories from opendap surface url"""

    full_url = url.format('dodsC')
    page = requests.get(full_url).text
    soup = BeautifulSoup(page, 'html.parser')
    dir_list = [url + '/' + node.get('href') for node in soup.find_all('a')
                if node.get('href').endswith('html')]
    new_list = []
    for path in dir_list:
        path_list = path.replace('//', '/').split('/')[1:]
        try:
            path_list.remove('catalog.html')
            dt.datetime.strptime(path_list[-1], '%Y%m%d%H')
            new_list.append(path_list[-1])
        except:
            continue
    return new_list
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def nco_exec(base_directory, site_name, date_directory, latitude, longitude):

    """Call the shell script that cuts out the site coordinates and
       concatenates with existing data (using NCO)"""

    exec_fpath = os.path.realpath(__file__)
    exec_dirpath = os.path.dirname(exec_fpath)
    shell_fpath = os.path.join(exec_dirpath, 'nco_shell.sh')
    exec_string = ('{0} "{1}" "{2}" "{3}" "{4}" "{5}"'
                   .format(shell_fpath, base_directory, site_name,
                           date_directory, latitude, longitude))
    if spc(exec_string, shell = True):
        error_str = ('Exception in NCO file processing - see NCO log file for '
                     'details')
        logging.error(error_str)
        raise RuntimeError('Error in command: {}'.format(exec_string))
    return
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def purge_dir(directory, file_ext = '.tmp'):

    """Dump any files not required"""

    f_list = filter(lambda x: os.path.splitext(x)[1] == '.tmp',
                    os.listdir(directory))
    for f in [os.path.join(directory, x) for x in f_list]: os.remove(f)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def wget_exec(read_path, write_path, server_dir):

    """Build the complete wget string and retrieve temp continental file"""

    logging.info('Retrieving forecast files for date {}'.format(server_dir))
    file_list = map(lambda x: '{}_{}'.format(server_dir, str(x).zfill(3)),
                    range(7))
    wget_log_path = os.path.join(base_dir, 'Log_files', 'Download.log')
    wget_prefix = '/usr/bin/wget -nv -a {} -O'.format(wget_log_path)
    for f in file_list:
        logging.info('Forecast +{} hrs'.format(str(int(f.split('_')[-1]))))
        tmp_fname = os.path.join(write_path, '{}_access.tmp'.format(f))
        full_read_path = read_path.format('fileServer') + server_dir
        server_fname = os.path.join(full_read_path,
                                    'ACCESS-R_{}_surface.nc'.format(f))
        cmd = '{0} {1} {2}'.format(wget_prefix, tmp_fname, server_fname)
        if spc(cmd, shell=True):
            error_str = ('Exception in wget command execution; see wget log '
                         'file for details')
            logging.error(error_str)
            raise RuntimeError(error_str)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### MAIN PROGRAM ###
#------------------------------------------------------------------------------

# Configure the logger
dt_str = dt.datetime.now().strftime("%Y%m%d%H%M")
full_log_path = os.path.join(base_log_path, 'ACCESS', 
                             'access_data_{}.log'.format(dt_str))
utils.set_logger(full_log_path)

# Get site details
site_df = utils.get_ozflux_site_list(master_file_path)

# Set and pre-purge requisite file paths
check_set_subdirs(base_dir)
continental_file_path = os.path.join(base_dir, 'Continental_files')
purge_dir(continental_file_path)

# Cross check available files on the opendap server against content of existing
# files
files_dict = check_seen_files(retrieval_path, base_dir, site_df.index)

# For each six-hour directory...
for this_dir in sorted(files_dict.keys()):

    # Grab the continent-wide files (n = 6)
    wget_exec(retrieval_path, continental_file_path, this_dir)

    # Cut out site from continent-wide file and append
    # (see shell script nco_shell.sh)
    sites_list = sorted(files_dict[this_dir])
    for site in sites_list:
        logging.info('Running site {}'.format(site))
        site_details = site_df.loc[site]
        nco_exec(base_dir, site_details.name.replace(' ',''), this_dir,
                 site_details.Latitude, site_details.Longitude)