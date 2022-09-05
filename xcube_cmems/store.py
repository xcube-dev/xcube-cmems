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
import xarray as xr
import logging
from typing import Any
from typing import List
from typing import Tuple
from typing import Container
from typing import Union
from typing import Iterator
from typing import Dict
from pydap.client import open_url
from xcube.core.gridmapping import GridMapping
from xcube.core.store import DataType
from xcube.core.store import DataStoreError
from xcube.core.store import DATASET_TYPE
from xcube.core.store import DataDescriptor
from .cmems import Cmems
from xcube.core.store import DataOpener
from xcube.core.store import DataStore
from xcube.core.store import DataTypeLike
from xcube.core.store import DatasetDescriptor
from xcube.core.store import VariableDescriptor
from xcube.util.assertions import assert_not_none
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonStringSchema
from xcube.util.jsonschema import JsonDateSchema
from xcube_cmems.constants import DATASET_OPENER_ID

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
    def _get_var_descriptors(xr_data_vars) -> Dict[str, VariableDescriptor]:
        var_descriptors = {}
        for var_key, var_value in xr_data_vars.variables.mapping.items():
            var_name = var_key
            var_dtype = var_value.dtype.type
            var_dims = var_value.dims
            if var_value.chunks:
                var_chunks = var_value.chunks
            else:
                var_chunks = None
            var_attrs = var_value.attrs
            var_descriptors[var_name] = \
                VariableDescriptor(name=var_name,
                                   dtype=var_dtype,
                                   dims=var_dims,
                                   chunks=var_chunks,
                                   attrs=var_attrs)
        return var_descriptors

    def describe_data(self, data_id: str) -> DatasetDescriptor:
        xr_ds = self.get_xarray_datastore()
        gm = GridMapping.from_dataset(xr_ds)
        attrs = xr_ds.attrs
        var_descriptors = self._get_var_descriptors(xr_ds.data_vars)
        coord_descriptors = self._get_var_descriptors(xr_ds.coords)
        # TODO : time_period
        temporal_coverage = (str(xr_ds.time[0].data).split('T')[0],
                             str(xr_ds.time[-1].data).split('T')[0])
        descriptor = DatasetDescriptor(data_id,
                                       data_type=self._data_type,
                                       crs=gm.crs,
                                       dims=xr_ds.dims,
                                       coords=coord_descriptors,
                                       data_vars=var_descriptors,
                                       attrs=attrs,
                                       bbox=gm.xy_bbox,
                                       spatial_res=gm.xy_res,
                                       time_range=temporal_coverage,
                                       # time_period=temporal_resolution
                                       )
        data_schema = self._get_open_data_params_schema(descriptor)
        descriptor.open_params_schema = data_schema
        return descriptor

    @staticmethod
    def get_pydap_datastore(url, session):
        return xr.backends.PydapDataStore(open_url(url, session=session,
                                                   user_charset='utf-8'))

    @property
    def copernicusmarine_datastore(self):
        urls = self.cmems.get_opendap_urls()
        try:
            _LOG.info(f'Getting pydap data store from {urls[0]}')
            data_store = self.get_pydap_datastore(urls[0], self.cmems.session)
        except AttributeError:
            _LOG.info(f'Getting data store from {urls[0]} failed,'
                      f'Now Getting pydap data store from {urls[1]}')
            data_store = self.get_pydap_datastore(urls[1], self.cmems.session)
        return data_store

    def get_xarray_datastore(self):
        pydap_ds = self.copernicusmarine_datastore
        return xr.open_dataset(pydap_ds)

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        # @TODO: TMH : use open_params
        assert_not_none(data_id)
        cmems_schema = self.get_open_data_params_schema(data_id)
        cmems_schema.validate_instance(open_params)
        return self.get_xarray_datastore()

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
            if dsd.crs.is_geographic:
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
            if return_tuples:
                if include_titles:
                    yield title, data_id
                else:
                    yield data_id, {}
            else:
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
        raise NotImplementedError("Search Data is not supported yet")

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
