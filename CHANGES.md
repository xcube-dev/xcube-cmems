## Changes in 0.1.6 (in development)

- Updated `pyproject.toml` file; package name changed from `xcube-cmems`
  to `xcube_cmems` and entry point removed, since xcube plugins 
  auto-recognition is updated. (#39 and xcube-dev/xcube#963)

- Support boolean-valued include_attrs in get_data_ids in accordance with API update in 
  xcube 1.8.0.

- Refactored `get_datasets_with_titles()` to align with the updated return type of 
 `cm.describe()`. (now returning a `CopernicusMarineCatalogue` object). The function 
  now accesses products and datasets via object attributes instead of dictionary keys.

- Updated dependency versions to ensure compatibility with `copernicusmarine` >= 2.1.1.

- Updated GitHub Actions workflow to use the latest Micromamba setup and use codecov.

## Changes in 0.1.5

- Disabled metadata cache to make it more suitable for cloud based environments. (#36)

## Changes in 0.1.4

- Changed initialization of cmems to not call copernicusmarine.login that writes a 
  configuration file. (#32)


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
