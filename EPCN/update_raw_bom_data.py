#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:07:48 2020

@author: imchugh

Function to write AWS data from BOM server to raw text file
"""

#------------------------------------------------------------------------------
### MODULES (STANDARD) ###
#------------------------------------------------------------------------------

import os
import sys

#------------------------------------------------------------------------------
### MODULES (CUSTOM) ###
#------------------------------------------------------------------------------

this_path = os.path.join(os.path.dirname(__file__), '../BOM_AWS')
sys.path.append(this_path)
import bom_functions as fbom
import utils

#------------------------------------------------------------------------------
### CONFIGURATIONS ###
#------------------------------------------------------------------------------

configs = utils.get_configs()
aws_file_path = configs['raw_data_write_paths']['bom'] 

#------------------------------------------------------------------------------
### MAIN PROGRAM ###
#------------------------------------------------------------------------------

# Get BOM data class and write ftp data to text file
aws = fbom.bom_data_getter()
aws.write_to_text_file(aws_file_path)