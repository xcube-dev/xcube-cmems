# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Any, List, Tuple, Container, Union, Iterator, Dict
import logging

import pyproj
import zarr
import xarray as xr
import numpy as np
import pandas as pd
from pydap.client import open_url
from pydap.model import DatasetType
from xarray.core.dataset import DataVariables
from xcube.core.gridmapping import GridMapping
from xcube.core.store import DataType
from xcube.core.store import DataStoreError
from xcube.core.store import DATASET_TYPE
from xcube.core.store import DataDescriptor
from xcube.core.store import DataOpener
from xcube.core.store import DataStore
from xcube.core.store import DataTypeLike
from xcube.core.store import DatasetDescriptor
from xcube.core.store import VariableDescriptor
from xcube.core.zarrstore import GenericZarrStore
from xcube.util.assertions import assert_not_none
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonStringSchema
from xcube.util.jsonschema import JsonDateSchema

from .constants import DATASET_OPENER_ID
from .constants import CAS_URL
from .constants import CSW_URL
from .constants import DATABASE
from .constants import ODAP_SERVER
from .default_env_vars import DEFAULT_CMEMS_USER
from .default_env_vars import DEFAULT_CMEMS_USER_PASSWORD
from .cmems import Cmems

_LOG = logging.getLogger('xcube')


class CmemsDataOpener(DataOpener):
    """
       Cmems implementation of the ``xcube.core.store.DataOpener``
       interface.
       """

    def __init__(self,
                 cmems: Cmems,
                 id: str,
                 data_type: DataType
                 ):
        self.cmems = cmems
        self._id = id
        self._data_type = data_type

    def dataset_names(self) -> List[str]:
        return self.cmems.dataset_names()

    def has_data(self, data_id: str) -> bool:
        return data_id in self.dataset_names()

    @staticmethod
    def _get_var_descriptors(xr_data_vars: DataVariables) -> \
            Dict[str, VariableDescriptor]:
        var_descriptors = {}
        for var_key, var_value in xr_data_vars.variables.mapping.items():
            var_name = var_key
            var_dtype = var_value.dtype.name
            var_dims = var_value.dims
            var_attrs = var_value.attrs
            var_descriptors[var_name] = \
                VariableDescriptor(name=var_name,
                                   dtype=var_dtype,
                                   dims=var_dims,
                                   attrs=var_attrs)
        return var_descriptors

    @staticmethod
    def _determine_time_period(data: xr.Dataset):
        if 'time' in data and len(data['time'].values) > 1:
            time_diff = data['time'].diff(
                dim=data['time'].dims[0]
            ).values.astype(np.float64)
            time_res = time_diff[0]
            time_regular = np.allclose(time_res, time_diff, 1e-8)
            if time_regular:
                time_period = pd.to_timedelta(time_res).isoformat()
                # remove leading P
                time_period = time_period[1:]
                # removing sub-day precision
                return time_period.split('T')[0]

    def describe_data(self, data_id: str) -> DatasetDescriptor:
        xr_ds = self.open_dataset(data_id)
        gm = GridMapping.from_dataset(xr_ds)
        attrs = xr_ds.attrs
        var_descriptors = self._get_var_descriptors(xr_ds.data_vars)
        coord_descriptors = self._get_var_descriptors(xr_ds.coords)
        temporal_resolution = self._determine_time_period(xr_ds)
        temporal_coverage = (str(xr_ds.time[0].data).split('T')[0],
                             str(xr_ds.time[-1].data).split('T')[0])
        descriptor = DatasetDescriptor(data_id,
                                       data_type=self._data_type,
                                       crs=gm.crs.name,
                                       dims=xr_ds.dims,
                                       coords=coord_descriptors,
                                       data_vars=var_descriptors,
                                       attrs=attrs,
                                       bbox=gm.xy_bbox,
                                       time_range=temporal_coverage,
                                       time_period=temporal_resolution)
        data_schema = self._get_open_data_params_schema(descriptor)
        descriptor.open_params_schema = data_schema
        return descriptor

    def get_pydap_dataset(self, data_id) -> DatasetType:
        urls = self.cmems.get_opendap_urls(data_id)
        open_url_kwargs = dict(
            session=self.cmems.session,
            user_charset='utf-8',  # CMEMS-specific
            output_grid=False  # retrieve only main arrays
        )
        try:
            _LOG.info(f'Getting pydap dataset from {urls[0]}')
            pyd_dataset = open_url(urls[0], **open_url_kwargs)
        except AttributeError:
            _LOG.info(f'Getting dataset from {urls[0]} failed,'
                      f'Now Getting pydap dataset from {urls[1]}')
            pyd_dataset = open_url(urls[1], **open_url_kwargs)
        return pyd_dataset

    def open_dataset(self, data_id) -> xr.Dataset:
        pydap_ds = self.get_pydap_dataset(data_id)
        global_attrs = dict(pydap_ds.attributes.get('NC_GLOBAL') or {})
        arrays = self.get_generic_arrays(pydap_ds)
        for array in arrays:
            if array["name"] == "time":
                if "chunks" in array:
                    del array["chunks"]
        max_cache_size: int = 2 ** 28
        zarr_store = GenericZarrStore(*arrays, attrs=global_attrs)
        if max_cache_size:
            zarr_store = zarr.LRUStoreCache(zarr_store, max_size=max_cache_size)
        dataset = xr.open_zarr(zarr_store)
        # Allow for accessing original zarr_store later (new in xcube 0.12.1)
        dataset.zarr_store.set(zarr_store)
        return dataset

    @classmethod
    def get_data(cls, pyd_var=None, chunk_info=None):
        array_slices = chunk_info["slices"]
        # Actual pydap data access
        if hasattr(pyd_var, "array"):
            data = pyd_var.array[array_slices]
        else:
            data = pyd_var[array_slices]
        return np.array(data)

    def get_generic_arrays(self, pyd_dataset) -> List[GenericZarrStore.Array]:
        """Get the list of generic arrays from the list of pydap
        variables in *pyd_dataset*.
        """
        arrays = []
        for name, pyd_var in pyd_dataset.items():
            attrs = dict(pyd_var.attributes)
            chunks = attrs.pop("_ChunkSizes", None)
            chunks = (chunks,) if isinstance(chunks, int) else \
                tuple(chunks) if chunks is not None else None
            fill_value = attrs.pop(
                "_FillValue",
                float('NaN') if np.issubdtype(pyd_var.dtype, np.floating)
                else None
            )
            array = GenericZarrStore.Array(
                name=name,
                dtype=pyd_var.dtype.str,
                dims=pyd_var.dimensions,
                shape=pyd_var.shape,
                chunks=chunks,
                fill_value=fill_value,
                get_data=self.get_data,
                get_data_params=dict(pyd_var=pyd_var),
                attrs=attrs,
                compressor=None,
                chunk_encoding="ndarray",
            )
            arrays.append(array)
        return arrays

    @staticmethod
    def subset_cube_with_open_params(ds: xr.Dataset,
                                     **open_params) -> xr.Dataset:
        if 'time_range' in open_params:
            ds = ds.sel(time=slice(open_params.get('time_range')[0],
                                   open_params.get('time_range')[1]))
        if 'bbox' in open_params:
            ds = ds.sel({"lat": slice(open_params.get('bbox')[1],
                                      open_params.get('bbox')[3]),
                         "lon": slice(open_params.get('bbox')[0],
                                      open_params.get('bbox')[2])})
        if 'variable_names' in open_params:
            ds = ds[open_params.get('variable_names')]
        return ds

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        assert_not_none(data_id)
        cmems_schema = self.get_open_data_params_schema(data_id)
        cmems_schema.validate_instance(open_params)
        open_params, other_kwargs = cmems_schema. \
            process_kwargs_subset(open_params, ('variable_names', 'time_range',
                                                'bbox'))
        ds = self.open_dataset(data_id)
        if open_params:
            ds = self.subset_cube_with_open_params(ds, **open_params)
        return ds

    def get_open_data_params_schema(self,
                                    data_id: str = None) -> JsonObjectSchema:
        if data_id is None:
            return self._get_open_data_params_schema()
        dsd = self.describe_data(data_id)
        return self._get_open_data_params_schema(dsd)

    @staticmethod
    def _get_open_data_params_schema(dsd: DatasetDescriptor = None) -> \
            JsonObjectSchema:
        min_date = dsd.time_range[0] if dsd and dsd.time_range else None
        max_date = dsd.time_range[1] if dsd and dsd.time_range else None
        dataset_params = dict(
            variable_names=JsonArraySchema(items=JsonStringSchema(
                enum=dsd.data_vars.keys() if dsd and dsd.data_vars else None)),
            time_range=JsonDateSchema.new_range(min_date, max_date)
        )
        if dsd:
            try:
                if pyproj.CRS.from_string(dsd.crs).is_geographic:
                    min_lon = dsd.bbox[0] if dsd and dsd.bbox else -180
                    min_lat = dsd.bbox[1] if dsd and dsd.bbox else -90
                    max_lon = dsd.bbox[2] if dsd and dsd.bbox else 180
                    max_lat = dsd.bbox[3] if dsd and dsd.bbox else 90
                    bbox = JsonArraySchema(items=(
                        JsonNumberSchema(minimum=min_lon, maximum=max_lon),
                        JsonNumberSchema(minimum=min_lat, maximum=max_lat),
                        JsonNumberSchema(minimum=min_lon, maximum=max_lon),
                        JsonNumberSchema(minimum=min_lat, maximum=max_lat)))
                    dataset_params['bbox'] = bbox
            except pyproj.exceptions.CRSError:
                pass
        cmems_schema = JsonObjectSchema(
            properties=dict(**dataset_params),
            required=[
            ],
            additional_properties=False
        )
        return cmems_schema


class CmemsDatasetOpener(CmemsDataOpener):

    def __init__(self, **cmems_params):
        super().__init__(
            Cmems(**cmems_params),
            DATASET_OPENER_ID,
            DATASET_TYPE,
        )


class CmemsDataStore(DataStore):
    """
       CMEMS implementation of the ``xcube.core.store.DataStore``
       interface.
       """

    def __init__(self, **store_params):
        cmems_schema = self.get_data_store_params_schema()
        cmems_schema.validate_instance(store_params)
        cmems_kwargs, store_params = cmems_schema.process_kwargs_subset(
            store_params, (
                'cmems_user',
                'cmems_user_password',
                'dataset_id',
                'cas_url',
                'csw_url',
                'databases',
                'server',
            ))
        self._dataset_opener = CmemsDatasetOpener(**cmems_kwargs)

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        cmems_params = dict(
            cmems_user=JsonStringSchema(
                title='CMEMS User',
                description='Preferably set by environment variable CMEMS_USER'
            ),
            cmems_user_password=JsonStringSchema(
                title='CMEMS User Password',
                description='Preferably set by environment '
                            'variable CMEMS_PASSWORD'
            ),
            cas_url=JsonStringSchema(default=CAS_URL),
            csw_url=JsonStringSchema(default=CSW_URL),
            databases=JsonStringSchema(default=DATABASE),
            server=JsonStringSchema(default=ODAP_SERVER),
        )
        required = None
        if not DEFAULT_CMEMS_USER or not DEFAULT_CMEMS_USER_PASSWORD:
            required = []
            if DEFAULT_CMEMS_USER is None:
                required.append('cmems_user')
            if DEFAULT_CMEMS_USER_PASSWORD is None:
                required.append('cmems_user_password')
        return JsonObjectSchema(
            properties=dict(**cmems_params),
            required=required,
            additional_properties=False
        )

    def open_data(self, data_id: str, opener_id: str = None,
                  **open_params) -> Any:
        return self._get_opener(opener_id=opener_id).open_data(data_id,
                                                               **open_params)

    def _get_opener(self, opener_id: str = None,
                    data_type: str = None) -> CmemsDataOpener:
        self._assert_valid_opener_id(opener_id)
        self._assert_valid_data_type(data_type)
        return self._dataset_opener

    def get_data_ids(self, data_type: DataTypeLike = None,
                     include_attrs: Container[str] = None) -> \
            Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        dataset_ids = self._dataset_opener.cmems.get_all_dataset_ids()
        return_tuples = include_attrs is not None
        include_titles = return_tuples and 'title' in include_attrs
        for data_id, title in dataset_ids.items():
            if include_titles:
                yield data_id, {'title': title}
            if return_tuples:
                yield data_id, {}
            yield data_id

    @classmethod
    def get_data_types(cls) -> Tuple[str, ...]:
        return DATASET_TYPE.alias,

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        return self.get_data_types()

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        return self._get_opener(data_type=data_type).has_data(data_id)

    def describe_data(self, data_id: str,
                      data_type: DataTypeLike = None) -> DataDescriptor:
        return self._get_opener(data_type=data_type).describe_data(data_id)

    def get_data_opener_ids(self, data_id: str = None,
                            data_type: DataTypeLike = None) -> Tuple[str, ...]:
        self._assert_valid_data_type(data_type)
        if data_id is not None \
                and not self.has_data(data_id, data_type=data_type):
            raise DataStoreError(f'Data resource {data_id!r}'
                                 f' is not available.')
        if data_type is not None \
                and not DATASET_TYPE.is_super_type_of(data_type):
            raise DataStoreError(f'Data resource {data_id!r}'
                                 f' is not available as type {data_type!r}.')
        return DATASET_OPENER_ID,

    def get_open_data_params_schema(self, data_id: str = None,
                                    opener_id: str = None) -> JsonObjectSchema:
        return self._get_opener(opener_id=opener_id). \
            get_open_data_params_schema(data_id)

    def search_data(self, data_type: DataTypeLike = None, **search_params) -> \
            Iterator[DataDescriptor]:
        raise NotImplementedError("search_data() operation is not "
                                  "supported yet")

    @classmethod
    def get_search_params_schema(cls,
                                 data_type: DataTypeLike = None) -> \
            JsonObjectSchema:
        pass

    ##########################################################################
    # Implementation helpers

    @classmethod
    def _is_valid_data_type(cls, data_type: str) -> bool:
        return data_type is None or DATASET_TYPE.is_super_type_of(data_type)

    @classmethod
    def _assert_valid_data_type(cls, data_type):
        if not cls._is_valid_data_type(data_type):
            raise DataStoreError(
                f'Data type must be {DATASET_TYPE!r},'
                f' but got {data_type!r}')

    @classmethod
    def _assert_valid_opener_id(cls, opener_id):
        if opener_id is not None and opener_id != DATASET_OPENER_ID:
            raise DataStoreError(f'Data opener identifier must be'
                                 f' "{DATASET_OPENER_ID}",'
                                 f' but got "{opener_id}"')
