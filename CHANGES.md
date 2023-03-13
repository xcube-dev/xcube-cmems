## Changes in 0.1.2 (in development)

- addressed initialization of cmems credentials from environment variables conditionally
  so that they are also evaluated when new_data_store is called.

## Changes in 0.1.1

- Fixed a bug when opening dataset with bounding box. Now the following spatial coords naming styles are
  supported: (lat,lon), (latitude,longitude), (y,x).
- set cookie 'CASTGC' optionally to avoid users from getting KeyError, closes #12
- addressed inconsisent parameter names for credentials and rely on global environment variables
  CMEMS_USERNAME and CMEMS_PASSWORD, closes  #11, #15 , #16, #17
- Use micromamba for CI.

## Changes in 0.1.0

First version of CMEMS Data Store.
