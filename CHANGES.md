## Changes in 0.1.2

- Improved the performance of get_data_ids method of the store by using asyncio and caching. (#18)


## Changes in 0.1.1

- Fixed a bug when opening dataset with bounding box. Now the following spatial coords naming styles are
  supported: (lat,lon), (latitude,longitude), (y,x).
- Set cookie 'CASTGC' only if it previously existed to avoid users from getting KeyError. (#12)

- Addressed inconsistent parameter names for credentials and rely on global environment variables
  CMEMS_USERNAME and CMEMS_PASSWORD.  (#11) (#15) (#16) (#17)

- Use micromamba for CI.

## Changes in 0.1.0

First version of CMEMS Data Store.
