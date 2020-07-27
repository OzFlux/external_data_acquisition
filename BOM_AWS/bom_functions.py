#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:48:26 2019

@author: ian
"""

# Standard modules
import datetime as dt
import ftplib
from io import BytesIO
import json
import math
import numpy as np
import os
import pandas as pd
from pytz import timezone
from pytz.exceptions import UnknownTimeZoneError
import requests
from time import sleep
from timezonefinder import TimezoneFinder as tzf
import zipfile
import pdb

#------------------------------------------------------------------------------
### REMOTE CONFIGURATIONS ###
#------------------------------------------------------------------------------

ftp_server = 'ftp.bom.gov.au'
ftp_dir = 'anon2/home/ncc/srds/Scheduled_Jobs/DS010_OzFlux/'

#------------------------------------------------------------------------------
### CLASSES ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class bom_data_getter(object):

    def __init__(self):

        self.stations = get_aws_station_details(with_timezone = True)
        self.file_format = get_data_file_formatting()
        self._zfile, self._empty_files = get_ftp_data(list_missing = True)

    #--------------------------------------------------------------------------
    def _check_line_integrity(self, line):

        """Check line length and number of elements are as expected, and that
           final character is #"""

        # Set values for validity checks
        line_len = 161
        element_n = 33

        # Do checks
        line_list = line.decode().split(',')
        try:
            assert len(line) == line_len # line length consistent?
            assert len(line_list) == element_n # number elements consistent?
            assert '#' in line_list[-1] # hash last character (ex carriage return)?
        except AssertionError:
            return False
        return True
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _generate_dummy_line(self, station_id, datetime):

        """Write out a line of dummy data with correct spacing and site
           details, but no met data"""

        station_details = self.stations.loc[station_id]
        start_list = ['dd', station_id, station_details.station_name.zfill(40)]
        try:
            local_datetime = self.get_local_datetime(datetime,
                                                     station_details.time_zone)
            local_datetime_str = dt.datetime.strftime(local_datetime,
                                                      '%Y,%m,%d,%H,%M')
        except UnknownTimeZoneError:
            local_datetime_str = '    ,  ,  ,  ,  '
        start_list.append(local_datetime_str)
        start_list.append(dt.datetime.strftime(datetime, '%Y,%m,%d,%H,%M'))
        blank_list = [' ' * x for x in self.file_format.Byte_length]
        new_str = ','.join((start_list + blank_list[5: -1] + ['#\r\n']))
        return self._set_line_order(new_str)
    #--------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    def get_dataframe(self, station_id):

        """Convert zipfile read from ftp site to dataframe (just index and
           data string) with dummy spaces"""

        readfile_name = self.get_file_id_dict()[station_id]
        header_line = 0
        with self._zfile.open(readfile_name) as zf:
            content = zf.readlines()
        valid_data = []
        for line in content[header_line + 1:]:
            if self._check_line_integrity(line):
                new_line = self._set_line_order(line)
                valid_data.append(new_line)
        if len(valid_data) == 0: return None
        dtstr_list = [','.join(x.split(',')[7: 12]) for x in valid_data]
        dt_list = [dt.datetime.strptime(x, '%Y,%m,%d,%H,%M') for x in dtstr_list]
        valid_df = pd.DataFrame(valid_data, index = dt_list, columns = ['Data'])
        new_index = pd.date_range(valid_df.index[0], valid_df.index[-1], freq = '30T')
        missing_dates = [x.to_pydatetime() for x in new_index
                         if not x in valid_df.index]
        dummy_data = [self._generate_dummy_line(station_id, x)
                      for x in missing_dates]
        dummy_df = pd.DataFrame(dummy_data, index = missing_dates, columns = ['Data'])
        df = pd.concat([valid_df, dummy_df])
        df.sort_index(inplace = True)
        return df
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_details_string(self, station_id):

        """Make a string with relevant details to be written into text file
           header"""

        name = self.stations.loc[station_id, 'station_name'].rstrip()
        state = self.stations.loc[station_id, 'State'].rstrip()
        name_state = '{0} ({1})'.format(name, state)
        lat = str(self.stations.loc[station_id, 'lat'])
        lon = str(self.stations.loc[station_id, 'lon'])
        coords = 'Coords: lat = {}, long = +{}'.format(lat, lon)
        elev = 'Elev: {}m asl'.format(self.stations.loc[station_id,
                                                        'height_stn_asl'].lstrip())
        return ', '.join([name_state, coords, elev]) + '\n'
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_file_id_dict(self):

        """Get a dictionary referencing the file name by key of BOM site ID"""

        data_list = [x for x in self._zfile.namelist() if 'Data' in x]
        return dict(zip([x.split('_')[2] for x in data_list], data_list))
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_header_string(self, as_list = False):

        """Get formatted header list for the output file"""

        headers = self.file_format.Description.tolist()
        new_headers = headers[:2] + headers[3:9] + headers[11:]
        new_headers[0] = new_headers[0].split('-')[0].rstrip()
        new_headers[1] = 'Station Number'
        new_headers[12] = new_headers[12].replace('km/h', 'm/s')
        new_headers[14] += ' true'
        new_headers[16] = new_headers[16].replace('km/h', 'm/s')
        new_headers[-1] = new_headers[-1].replace(',', ' ')
        if as_list: return new_headers
        new_header_str = ','.join(new_headers) + '\n'
        return new_header_str
    #--------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    def get_local_datetime(self, dt_obj, tz_name):

        """Convert standard local datetime to local datetime using timezone"""

        tz_obj = timezone(tz_name)
        try:
            dst_offset = tz_obj.dst(dt_obj)
        except:
            dst_offset = tz_obj.dst(dt_obj + dt.timedelta(seconds = 3600))
        return dt_obj + dst_offset
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _get_write_list(self, station_id, file_dir):

        """Compare the local file content with the content from the ftp server,
           and keep all lines that are missing from the local (add dummy lines)
           if there is a gap"""

        # Get ftp data
        ftp_content = self.get_dataframe(station_id)

        # Get local data
        local_filepath = os.path.join(file_dir,
                                      'HM01X_Data_{}.txt'.format(station_id))
        try:
            with open(local_filepath) as f:
                print('Existing file found in path; appending new remote data...')
                local_content = f.readlines()
        except FileNotFoundError:
            print ('No local file found; writing all available '
                   'remote data to local directory...'.format(station_id))
            return ([self.get_details_string(station_id)] +
                    [self.get_header_string()] + ftp_content.Data.tolist())

        # Retain remote data not found in local file and pad dummy cases if
        # there is a gap
        last_date_str = ','.join(local_content[-1].split(',')[7:12])
        last_date = dt.datetime.strptime(last_date_str, '%Y,%m,%d,%H,%M')
        if last_date == ftp_content.index[-1].to_pydatetime():
            print ('File already up to date!')
            return
        try:
            int_loc = ftp_content.index.get_loc(last_date)
            write_list = ftp_content.iloc[int_loc + 1:]['Data'].tolist()
        except KeyError:
            date_range = (pd.date_range(last_date, ftp_content.index[0],
                                        freq = '30T')
                          .to_pydatetime())[1:-1]
            write_list = [self._generate_dummy_line(station_id, x)
                          for x in date_range] + ftp_content.Data.tolist()
        return write_list
    #------------------------------------------------------------------------------

    #------------------------------------------------------------------------------
    def report_empty_files(self, station_names = True):

        """Return list of files that contained no data"""

        id_list = sorted([x.split('_')[2] for x in self._empty_files])
        if not station_names: return id_list
        return list(zip(id_list, [self.stations.loc[i, 'station_name'].rstrip()
                                  for i in id_list]))
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _set_line_order(self, line):

        """Drop wet bulb temperature and convert wind speed"""

        def convert_kmh_2_ms(kmh):
            try:
                return str(round(float(kmh) / 3.6, 1)).rjust(5)
            except:
                return kmh.rjust(5)
        if isinstance(line, str): line_list = line.split(',')
        if isinstance(line, bytes): line_list = line.decode().split(',')
        line_list[1] = line_list[1].zfill(6)
        line_list[23] = convert_kmh_2_ms(line_list[23])
        line_list[27] = convert_kmh_2_ms(line_list[27])
        return ','.join(line_list[:2] + line_list[3:17] + line_list[19:])
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def write_to_text_file(self, write_path, station_id = None):

        """Write data to text file (append only new data if file exists)"""

        if station_id:
            station_list = [station_id]
        else:
            station_list = sorted(self.get_file_id_dict().keys())
        for this_station in station_list:
            print ('Updating file for station ID {}'.format(this_station))
            local_filepath = os.path.join(write_path,
                                          'HM01X_Data_{}.txt'.format(this_station))
            data_to_write = self._get_write_list(this_station, write_path)
            if not data_to_write: continue
            with open(local_filepath, 'a+') as f:
                f.writelines(data_to_write)
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------
### FUNCTIONS ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_aws_station_details(with_timezone = False):

    """Retrieves a dataframe containing details of all AWS stations on the
       OzFlux ftp server site"""

    zf = get_ftp_data(['StnDet'], get_first = False)
    data_list = []
    for f in zf.namelist():
        with zf.open(f) as file_obj:
            for line in file_obj:
                data_list += [line.decode().split(',')]
    notes_df = get_station_details_formatting()
    df = pd.DataFrame(data_list, columns = notes_df.Description)
    for col in [x for x in df.columns if not 'station_id' in x]:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            continue
    df.index = df['station_id']
    df['station_name'] = [x.rstrip() for x in df.station_name]
    df = df[~df.index.duplicated(keep='first')]
    if with_timezone:
        df['time_zone'] = [get_timezone(df.loc[idx, 'lat'], df.loc[idx, 'lon'])
                           for idx in df.index]
    return df.sort_index()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_data_file_formatting():

    """Gets the file format for the data files on the OzFlux-specific ftp
       server"""

    zf = get_ftp_data(search_list = ['Notes'])
    with zf.open(zf.namelist()[0]) as file_obj:
        notes_list = file_obj.readlines()
    bad_chars = '*_-.'
    start_line = 19
    end_line = 44
    byte_start = ['Byte_start']
    byte_length = ['Byte_length']
    desc = ['Description']
    for i, line in enumerate(notes_list[start_line:end_line]):
        line_list = [x.strip() for x in line.decode().split(',')]
        byte_start.append(int(line_list[0].split('-')[0]))
        byte_length.append(int(line_list[1]))
        if i > 1: line_list[2] = line_list[2].translate({ord(c): None for c in
                                                         bad_chars})
        if len(line_list) > 2: line_list[2] = ','.join(line_list[2:])
        desc.append(line_list[2].rstrip('.').lstrip())
    return pd.DataFrame({byte_start[0]: byte_start[1:],
                         byte_length[0]: byte_length[1:],
                         desc[0]: desc[1:]})
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_ftp_data(search_list = None, get_first = True, list_missing = False):

    """Function to retrieve zipfile containing data files for AWS stations
       on OzFlux ftp server site
       Args:
           * search_list (list or None, default None): specifies substrings
             used to search for matching file names (if None returns all data
             available)
           * get_first (boolean, default True): if true, returns only the first
             file with matching substring"""

    # Login to ftp server
    ftp = ftplib.FTP(ftp_server)
    ftp.login()

    # Open the separate zip files and combine in a single zip file
    # held in dynamic memory - ignore the solar data
    missing_list = []
    master_bio = BytesIO()
    master_zf = zipfile.ZipFile(master_bio, 'w')
    zip_file_list = [os.path.split(f)[1] for f in ftp.nlst(ftp_dir)]
    for this_file in zip_file_list:
        if 'globalsolar' in this_file: continue
        in_file = os.path.join(ftp_dir, this_file)
        f_str = 'RETR {0}'.format(in_file)
        bio = BytesIO()
        ftp.retrbinary(f_str, bio.write)
        bio.seek(0)
        zf = zipfile.ZipFile(bio)
        if not search_list is None:
            file_list = []
            for this_str in search_list:
                if not isinstance(this_str, str):
                    raise TypeError('search_list elements must be of type str!')
                for f in zf.namelist():
                    if this_str in f:
                        file_list.append(f)
                        if get_first: search_list.remove(this_str)
        else:
            file_list = zf.namelist()
        for f in file_list:
            if not f in master_zf.namelist():
                temp = zf.read(f)
                if temp:
                    master_zf.writestr(f, temp)
                else:
                    missing_list.append(f)
                    print ('No data in file {}! Skipping...'.format(f))
        zf.close()
        if not search_list is None and len(search_list) == 0: break
    ftp.close()

    if list_missing and len(missing_list) > 0 : return master_zf, missing_list
    return master_zf
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_nearest_bom_station(lat, lon, start_date = None, end_date = None,
                            nearest_n = 5):

    """Uses haversine formula to estimate distance from given coordinates
       to nearest x stations"""

    try: py_start_date = dt.datetime.strptime(start_date, '%Y%m%d').date()
    except TypeError: None
    try: py_end_date = dt.datetime.strptime(end_date, '%Y%m%d').date()
    except TypeError: py_end_date = dt.datetime.now().date()
    stations = get_aws_station_details(with_timezone=True)
    if start_date:
        py_start = []
        for st in stations.index:
            try:
                py_start.append(dt.datetime.strptime(stations.loc[st, 'month_year_opened'],
                                                     '%m/%Y').date())
            except ValueError: py_start.append(None)
        stations = stations.loc[np.array(py_start) <= py_start_date]
    if end_date:
        py_end = []
        for st in stations.index:
            try:
                py_end.append(dt.datetime.strptime(stations.loc[st, 'month_year_closed'],
                                                   '%m/%Y').date())
            except ValueError: py_end.append(dt.datetime.now().date())
        stations = stations.loc[np.array(py_end) >= py_end_date]
    stations['dist (km)'] = list(map(lambda x: haversine(lat, lon,
                                                         stations.loc[x, 'lat'],
                                                         stations.loc[x, 'lon']),
                                     stations.index))
    df = stations.sort_values(['dist (km)']).head(nearest_n)
    return df[['station_name', 'lat', 'lon', 'month_year_opened', 
               'month_year_closed', 'dist (km)', 'time_zone']]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_station_details_formatting(truncate_description = True):

    """Gets the file format for the station details file on the ozflux-specific
       ftp server"""

    simple_list = ['record_id', 'station_id', 'rainfall_district_id',
                    'station_name', 'month_year_opened', 'month_year_closed',
                    'lat', 'lon', 'coords_derivation', 'State',
                    'height_stn_asl', 'height_barom_asl', 'wmo_id',
                    'first_file_year', 'last_file_year', 'pct_complete',
                    'pct_vals_Y', 'pct_vals_N', 'pct_vals_W', 'pct_vals_S',
                    'pct_vals_I', 'eor']
    start_line = 408
    end_line = 430
    zf = get_ftp_data(search_list = ['Notes'])
    with zf.open(zf.namelist()[0]) as file_obj:
        notes_list = file_obj.readlines()
    byte_start = ['Byte_start']
    byte_length = ['Byte_length']
    desc = ['Description']
    if truncate_description: desc += simple_list
    for line in notes_list[start_line:end_line]:
        line_list = [x.strip() for x in line.decode().split(',')]
        byte_start.append(int(line_list[0].split('-')[0]))
        byte_length.append(int(line_list[1]))
        if not truncate_description: desc.append(line_list[2].rstrip())
    return pd.DataFrame({byte_start[0]: byte_start[1:],
                         byte_length[0]: byte_length[1:],
                         desc[0]: desc[1:]})
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_timezone(lat, lon):

    """Get timezone for coordinates"""

    def make_tz_request(lat, lon):

        api_key = 'UT66CKBKU8MM'
        base_url_str = 'http://api.timezonedb.com/v2.1/get-time-zone'
        end_str = ('?key={0}&format=json&by=position&lat={1}&lng={2}'
                   .format(api_key, lat, lon))
        sleep(1)
        json_obj = requests.get(base_url_str + end_str)
        if json_obj.status_code == 200:
            return json.loads(json_obj.content)
        else:
            print ('Timezone request returned status code {}'
                   .format(json_obj.status_code))

    # Create a timezone variable using lookup from python package, fall back
    # on API
    try:
        tz = tzf().timezone_at(lng = lon, lat = lat)
    except TypeError:
        pdb.set_trace()
    if tz: return tz
    tz = make_tz_request(lat, lon)['zoneName']
    if tz: return tz
    return None
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):

    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat / 2)**2 + math.cos(lat1) *
         math.cos(lat2) * math.sin(dlon / 2)**2)
    r = 6371 # Radius of earth in kilometers
    c = 2 * math.asin(math.sqrt(a))
    return c * r
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### MAIN PROGRAM ###
#------------------------------------------------------------------------------

#if __name__ == "__main__":

    

#------------------------------------------------------------------------------
#def get_header_list():
#
#    l = ['hm',
#         'Station Number',
#         'Year Month Day Hour Minutes in YYYY,MM,DD,HH24,MI format in Local time',
#         'Year Month Day Hour Minutes in YYYY,MM,DD,HH24,MI format in Local standard time',
#         'Precipitation since 9am local time in mm',
#         'Quality of precipitation since 9am local time',
#         'Air Temperature in degrees C',
#         'Quality of air temperature',
#         'Dew point temperature in degrees C',
#         'Quality of dew point temperature',
#         'Relative humidity in percentage %',
#         'Quality of relative humidity',
#         'Wind speed in m/s',
#         'Wind speed quality',
#         'Wind direction in degrees true',
#         'Wind direction quality',
#         'Speed of maximum windgust in last 10 minutes in m/s',
#         'Quality of speed of maximum windgust in last 10 minutes',
#         'Station level pressure in hPa',
#         'Quality of station level pressure',
#         'AWS Flag',
#         '#\r\n']
#
#    return [','.join(l)]

#------------------------------------------------------------------------------

##------------------------------------------------------------------------------
#def get_ftp_files():
#
#    ftp = ftplib.FTP(ftp_server)
#    ftp.login()
#    zip_file_list = [os.path.split(f)[1] for f in ftp.nlst(ftp_dir)]
#    d = {}
#    for fname in zip_file_list:
#        in_file = os.path.join(ftp_dir, fname)
#        f_str = 'RETR {0}'.format(in_file)
#        bio = BytesIO()
#        ftp.retrbinary(f_str, bio.write)
#        bio.seek(0)
#        zf = zipfile.ZipFile(bio)
#        d[fname] = zf.namelist()
#        zf.close()
#    return d
##------------------------------------------------------------------------------