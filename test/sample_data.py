import numpy as np
import pandas as pd
import xarray as xr


def create_cmems_dataset():
    lat = xr.DataArray(
        np.linspace(0.0, 1.0, 5),
        dims=["lat"],
        attrs=dict(
            standard_name="latitude",
            long_name="latitude coordinate",
            units="degrees_north",
            axis="Y",
            step=0.08333588,
        ),
    )

    lon = xr.DataArray(
        np.linspace(0.0, 1.0, 5),
        dims=["lon"],
        attrs=dict(
            standard_name="longitude",
            long_name="longitude_coordinate",
            units="degrees_east",
            axis="X",
            step=0.08332825,
        ),
    )
    time = np.array(pd.date_range(start="1/1/2022", periods=8), dtype=np.datetime64)

    VHM0 = xr.DataArray(
        data=np.zeros((8, 5, 5)),
        dims=["time", "lat", "lon"],
        attrs=dict(
            long_name="Spectral significant wave height (Hm0)",
            units="m",
            standard_name="sea_surface_wave_significant_height",
            cell_methods="time:point area:mean",
            type_of_analysis="spectral analysis",
            WMO_code=100,
        ),
    )

    VTPK = xr.DataArray(
        data=np.zeros((8, 5, 5)),
        dims=["time", "lat", "lon"],
        attrs=dict(
            long_name="Spectral significant wave height (Hm0)",
            units="m",
            standard_name="sea_surface_wave_significant_height",
            cell_methods="time:point area:mean",
            type_of_analysis="spectral analysis",
            WMO_code=100,
        ),
    )

    return xr.Dataset(
        dict(VHM0=VHM0, VTPK=VTPK),
        coords=dict(time=time, lon=lon, lat=lat),
        attrs={
            "title": "Mean fields from global wave model"
            " MFWAM of Meteo-France with ECMWF forcing",
            "conventions": "CF-1.6",
            "institution": "METEO-FRANCE",
            "product_type": "hindcast",
            "product": "GLOBAL_ANALYSIS_FORECAST_WAV_001_027",
            "product_ref_date": "20220829-00:00:00",
            "product_range": "D-1",
            "product_user_manual": "http://marine.copernicus.eu/"
            "documents/PUM/CMEMS-GLO-PUM-"
            "001-027.pdf",
            "dataset": "global-analysis-forecast-wav-001-027",
            "references": "http://marine.copernicus.eu",
            "credit": "E.U. Copernicus Marine Service Information" " (CMEMS)",
            "licence": "http://marine.copernicus.eu/services-"
            "portfolio/service-commitments-and",
            "contact": "servicedesk.cmems@mercator-ocean.eu",
            "producer": "CMEMS - Global Monitoring and " "Forecasting Centre",
            "area": "GLO",
            "geospatial_lon_min": -180.0,
            "geospatial_lon_max": 179.9167,
            "geospatial_lon_step": 0.08332825,
            "geospatial_lon_units": "degree",
            "geospatial_lat_min": -80.0,
            "geospatial_lat_max": 90.0,
            "geospatial_lat_step": 0.08333588,
            "geospatial_lat_units": "degree",
            "time_coverage_start": "2022-01-01T10:59:38.888000Z",
            "time_coverage_end": "2022-01-08T10:59:38.888000Z",
            "crs": {"name": "EPSG:4326"},
        },
    )
