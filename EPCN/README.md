# external_data_retrieval
This repo contains the scripts that capture data from Bureau of Meteorology (ACCESS and AWS) and MODIS (several bands and products) servers. In the case of the BOM data streams, the data are continuously concatenated to existing raw files. In the case of the AWS files, the data for all available BOM AWS sites is retained (with the exception of some classified sites e.g. military bases). In the case of the ACCESS files, a 3x3 tile cut out from the continent-wide ACCESS runs is appended for each site in the OzFlux network (defined by a site details file not archived here). From these raw data files, data are processed to netcdf files formatted to plug into the PyFluxPro datastream for each site. MODIS data are read from the server and written directly to formatted file with no intervening step.

Note that the BOM_functions and modis_functions_rest modules (held in separate repos) are required for file updating. ACCESS updating scripts are self-contained.

The directory structure is defined by the paths.init file. 

