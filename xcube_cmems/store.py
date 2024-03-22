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
import xarray as xr
import numpy as np
import pandas as pd
import copernicusmarine as cm

from xarray.core.dataset import DataVariables
from xcube.core.gridmapping import GridMapping
from xcube.core.store import (
    DATASET_TYPE,
    DataDescriptor,
    DataOpener,
    DataStore,
    DataStoreError,
    DataType,
    DataTypeLike,
    DatasetDescriptor,
    VariableDescriptor,
)
from xcube.util.assertions import assert_not_none
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonDateSchema,
    JsonNumberSchema,
    JsonObjectSchema,
    JsonStringSchema,
)

from .cmems import Cmems
from .constants import DATASET_OPENER_ID

_LOG = logging.getLogger("xcube")


class CmemsDataOpener(DataOpener):
    """
    Cmems implementation of the ``xcube.core.store.DataOpener``
    interface.
    """

    def __init__(self, cmems: Cmems, id: str, data_type: DataType):
        self.cmems = cmems
        self._id = id
        self._data_type = data_type

    def dataset_names(self) -> List[dict]:
        return self.cmems.get_datasets_with_titles()

    def has_data(self, data_id: str) -> bool:
        return data_id in self.dataset_names()

    @staticmethod
    def _get_var_descriptors(
        xr_data_vars: DataVariables,
    ) -> Dict[str, VariableDescriptor]:
        var_descriptors = {}
        for var_key, var_value in xr_data_vars.variables.mapping.items():
            var_name = var_key
            var_dtype = var_value.dtype.name
            var_dims = var_value.dims
            var_attrs = var_value.attrs
            var_descriptors[var_name] = VariableDescriptor(
                name=var_name, dtype=var_dtype, dims=var_dims, attrs=var_attrs
            )
        return var_descriptors

    @staticmethod
    def _determine_time_period(data: xr.Dataset):
        if "time" in data and len(data["time"].values) > 1:
            time_diff = (
                data["time"].diff(dim=data["time"].dims[0]).values.astype(np.float64)
            )
            time_res = time_diff[0]
            time_regular = np.allclose(time_res, time_diff, 1e-8)
            if time_regular:
                time_period = pd.to_timedelta(time_res).isoformat()
                # remove leading P
                time_period = time_period[1:]
                # removing sub-day precision
                return time_period.split("T")[0]

    def describe_data(self, data_id: str) -> DatasetDescriptor:
        xr_ds = self.cmems.open_dataset(data_id)
        gm = GridMapping.from_dataset(xr_ds)
        attrs = xr_ds.attrs
        var_descriptors = self._get_var_descriptors(xr_ds.data_vars)
        coord_descriptors = self._get_var_descriptors(xr_ds.coords)
        temporal_resolution = self._determine_time_period(xr_ds)
        temporal_coverage = (
            str(xr_ds.time[0].data).split("T")[0],
            str(xr_ds.time[-1].data).split("T")[0],
        )
        descriptor = DatasetDescriptor(
            data_id,
            data_type=self._data_type,
            crs=gm.crs.name,
            dims=xr_ds.dims,
            coords=coord_descriptors,
            data_vars=var_descriptors,
            attrs=attrs,
            bbox=gm.xy_bbox,
            time_range=temporal_coverage,
            time_period=temporal_resolution,
        )
        data_schema = self.get_open_data_params_schema()
        descriptor.open_params_schema = data_schema
        return descriptor

    @staticmethod
    def subset_cube_with_open_params(dataset_id, **open_params) -> xr.Dataset:
        params = {}

        # Process time_range if present
        if "time_range" in open_params:
            params["start_datetime"], params["end_datetime"] = open_params["time_range"]

        # Process bbox if present
        if "bbox" in open_params:
            (
                params["minimum_longitude"],
                params["minimum_latitude"],
                params["maximum_longitude"],
                params["maximum_latitude"],
            ) = open_params["bbox"]

        # Process variable_names if present
        if "variable_names" in open_params:
            params["variables"] = open_params["variable_names"]

        return cm.open_dataset(dataset_id=dataset_id, **params)

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        assert_not_none(data_id)
        cmems_schema = self.get_open_data_params_schema(data_id)
        cmems_schema.validate_instance(open_params)
        open_params, other_kwargs = cmems_schema.process_kwargs_subset(
            open_params, ("variable_names", "time_range", "bbox")
        )
        return self.subset_cube_with_open_params(data_id, **open_params)

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        dataset_params = dict(
            variable_names=JsonArraySchema(
                items=(JsonStringSchema(min_length=0)), unique_items=True
            ),
            time_range=JsonDateSchema.new_range(),
            bbox=JsonArraySchema(
                items=(
                    JsonNumberSchema(minimum=-180, maximum=180),
                    JsonNumberSchema(minimum=-90, maximum=90),
                    JsonNumberSchema(minimum=-180, maximum=180),
                    JsonNumberSchema(minimum=-90, maximum=90),
                )
            ),
        )
        cmems_schema = JsonObjectSchema(
            properties=dict(**dataset_params), required=[], additional_properties=False
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
            store_params, ("cmems_username", "cmems_password")
        )
        self._dataset_opener = CmemsDatasetOpener(**cmems_kwargs)

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        cmems_params = dict(
            cmems_username=JsonStringSchema(
                title="CMEMS Username",
                description="Preferably set by environment variable " "CMEMS_USERNAME ",
            ),
            cmems_password=JsonStringSchema(
                title="CMEMS User Password",
                description="Preferably set by environment " "variable CMEMS_PASSWORD",
            ),
        )
        return JsonObjectSchema(
            properties=dict(**cmems_params), required=None, additional_properties=False
        )

    def open_data(self, data_id: str, opener_id: str = None, **open_params) -> Any:
        return self._get_opener(opener_id=opener_id).open_data(data_id, **open_params)

    def _get_opener(
        self, opener_id: str = None, data_type: str = None
    ) -> CmemsDataOpener:
        self._assert_valid_opener_id(opener_id)
        self._assert_valid_data_type(data_type)
        return self._dataset_opener

    def get_data_ids(
        self, data_type: DataTypeLike = None, include_attrs: Container[str] = None
    ) -> Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        dataset_ids_with_titles = self._dataset_opener.cmems.get_datasets_with_titles()
        return_tuples = include_attrs is not None
        include_titles = return_tuples and "title" in include_attrs

        for dataset in dataset_ids_with_titles:
            data_id = dataset["dataset_id"]
            if return_tuples:
                attrs = {}
                if include_titles:
                    attrs["title"] = dataset["title"]
                yield data_id, attrs
            else:
                yield data_id

    @classmethod
    def get_data_types(cls) -> Tuple[str, ...]:
        return (DATASET_TYPE.alias,)

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        return self.get_data_types()

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        return self._get_opener(data_type=data_type).has_data(data_id)

    def describe_data(
        self, data_id: str, data_type: DataTypeLike = None
    ) -> DataDescriptor:
        return self._get_opener(data_type=data_type).describe_data(data_id)

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> Tuple[str, ...]:
        self._assert_valid_data_type(data_type)
        if data_id is not None and not self.has_data(data_id, data_type=data_type):
            raise DataStoreError(f"Data resource {data_id!r}" f" is not available.")
        if data_type is not None and not DATASET_TYPE.is_super_type_of(data_type):
            raise DataStoreError(
                f"Data resource {data_id!r}" f" is not available as type {data_type!r}."
            )
        return (DATASET_OPENER_ID,)

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        return self._get_opener(opener_id=opener_id).get_open_data_params_schema(
            data_id
        )

    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[DataDescriptor]:
        raise NotImplementedError("search_data() operation is not " "supported yet")

    @classmethod
    def get_search_params_schema(
        cls, data_type: DataTypeLike = None
    ) -> JsonObjectSchema:
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
                f"Data type must be {DATASET_TYPE!r}," f" but got {data_type!r}"
            )

    @classmethod
    def _assert_valid_opener_id(cls, opener_id):
        if opener_id is not None and opener_id != DATASET_OPENER_ID:
            raise DataStoreError(
                f"Data opener identifier must be"
                f' "{DATASET_OPENER_ID}",'
                f' but got "{opener_id}"'
            )
