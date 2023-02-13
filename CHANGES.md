## Changes in 0.1.2

- Improved the performance of get_data_ids method of the store by using asyncio and caching,closes #18

## Changes in 0.1.1

- Fixed a bug when opening dataset with bounding box. Now the following spatial coords naming styles are
  supported: (lat,lon), (latitude,longitude), (y,x).
- Use micromamba for CI.

## Changes in 0.1.0

First version of CMEMS Data Store.
