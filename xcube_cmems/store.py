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
from typing import Any, Tuple, Container, Union, Iterator, Dict

import pyproj
import xarray as xr
from pydap.client import open_url

from xcube.core.store import DataDescriptor
from .cmems import Cmems
from xcube.core.store import DataOpener
from xcube.core.store import DataStore
from xcube.core.store import DataTypeLike
from xcube.core.store import DatasetDescriptor
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonStringSchema
from xcube.util.jsonschema import JsonDateSchema


class CmemsDataOpener(DataOpener):
    """
       Cmems implementation of the ``xcube.core.store.DataOpener``
       interface.

       Please refer to the :math:open_data method for the list of possible
       open parameters.
       """

    def __init__(self,
                 cmems: Cmems,
                 id: str,
                 ):
        self.cmems = cmems
        self._id = id

    def describe_data(self, data_id: str, data_type: DataTypeLike = None) \
            -> DatasetDescriptor:
        pass
        # ds = self.open_data(data_id)
        #
        # descriptor = DatasetDescriptor(data_id,
        #                                data_type=self._data_type,
        #                                crs=crs,
        #                                dims=dims,
        #                                coords=coord_descriptors,
        #                                data_vars=var_descriptors,
        #                                attrs=attrs,
        #                                bbox=bbox,
        #                                spatial_res=spatial_resolution,
        #                                time_range=temporal_coverage,
        #                                time_period=temporal_resolution)
        # data_schema = self._get_open_data_params_schema(descriptor)
        # descriptor.open_params_schema = data_schema
        # return descriptor

    @staticmethod
    def get_pydap_datastore(url, session):
        return xr.backends.PydapDataStore(open_url(url, session=session,
                                                   user_charset='utf-8'))

    @property
    def copernicusmarine_datastore(self):
        urls = self.cmems.get_opendap_urls()
        try:
            data_store = self.get_pydap_datastore(urls[0], self.cmems.session)
        except AttributeError:
            data_store = self.get_pydap_datastore(urls[1], self.cmems.session)

        return data_store

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        # @TODO: Remove the block comment later
        # assert_not_none(data_id)
        # cmems_schema = self.get_open_data_params_schema(data_id)
        # cmems_schema.validate_instance(open_params)
        # cube_kwargs, open_params = cmems_schema.process_kwargs_subset(
        #     open_params, ('variable_names', 'time_range', 'bbox'))
        data_store = self.copernicusmarine_datastore
        ds = xr.open_dataset(data_store)
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
            normalize_data=JsonBooleanSchema(default=True),
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
                # do not set bbox then
                pass
        cmems_schema = JsonObjectSchema(
            properties=dict(**dataset_params),
            required=[
            ],
            additional_properties=False
        )
        return cmems_schema


class CmemsDataStore(DataStore):

    def __init__(self, cmems, dataset_id, **store_params):
        self._dataset_opener = CmemsDataOpener(cmems, dataset_id)

    def search_data(self, data_type: DataTypeLike = None, **search_params) -> \
            Iterator[DataDescriptor]:
        pass

    @classmethod
    def get_search_params_schema(cls,
                                 data_type: DataTypeLike = None) -> \
            JsonObjectSchema:
        pass

    @classmethod
    def get_data_types(cls) -> Tuple[str, ...]:
        pass

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        pass

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        pass

    def describe_data(self, data_id: str,
                      data_type: DataTypeLike = None) -> DataDescriptor:
        pass

    def get_data_opener_ids(self, data_id: str = None,
                            data_type: DataTypeLike = None) -> Tuple[str, ...]:
        pass

    def get_open_data_params_schema(self, data_id: str = None,
                                    opener_id: str = None) -> JsonObjectSchema:
        pass

    def open_data(self, data_id: str, opener_id: str = None,
                  **open_params) -> Any:
        pass

    def _get_opener(self, opener_id: str = None,
                    data_type: str = None) -> CmemsDataOpener:
        # self._assert_valid_opener_id(opener_id)
        # self._assert_valid_data_type(data_type)
        return self._dataset_opener

    def get_data_ids(self, data_type: DataTypeLike = None,
                     include_attrs: Container[str] = None) -> \
            Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        dataset_ids = self._dataset_opener.cmems.get_all_dataset_ids()
        return_tuples = include_attrs is not None
        include_titles = return_tuples and 'title' in include_attrs
        for data_id, title in dataset_ids.values():
            if return_tuples:
                if include_titles:
                    yield title, data_id
                else:
                    yield data_id, {}
            else:
                yield data_id
