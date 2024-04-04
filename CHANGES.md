## Changes in 0.1.4

- Changed initialization of cmems to not call copernicusmarine.login that writes a 
  configuration_file. (#32)

## Changes in 0.1.3

- Changed the implementation to use the new cmems toolbox api (#31)

- Moved project configuration to `pyproject.toml`

## Changes in 0.1.2 

- Addressed initialization of cmems credentials from environment variables conditionally
  so that they are evaluated when new_data_store is called. (#21)

- Revised get_open_data_params logic to not open the dataset but only return 
  open_params without actual metadata. (#26)

- Improved the performance of get_data_ids method of the store by using 
  package `aiohttp` to make parallel calls to CSW API. (#18)


## Changes in 0.1.1

- Fixed a bug when opening dataset with bounding box. Now the following spatial coords naming 
  styles are supported: (lat,lon), (latitude,longitude), (y,x).

- Set cookie 'CASTGC' only if it previously existed to avoid users from getting KeyError. (#12)

- Addressed inconsisent parameter names for credentials and rely on global environment 
  variables CMEMS_USERNAME and CMEMS_PASSWORD. (#11) (#15) (#16) (#17)

- Use micromamba for CI.

## Changes in 0.1.0

First version of CMEMS Data Store.
