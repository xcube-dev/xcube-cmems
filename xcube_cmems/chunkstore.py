import copy
from abc import ABC
import itertools
import json
import math
import time
from abc import abstractmethod, ABCMeta
from collections.abc import MutableMapping
from typing import Iterator, Any, List, Dict, Tuple, Callable, \
    Iterable, KeysView, Mapping

import numpy as np
import pandas as pd
# import pyproj
from numcodecs import Blosc

# from .config import CubeConfig
# from .constants import BAND_DATA_ARRAY_NAME
# from .constants import CRS_ID_TO_URI
from xcube_cmems import cmems

_STATIC_ARRAY_COMPRESSOR_PARAMS = dict(
    cname='zstd',
    clevel=1,
    shuffle=Blosc.SHUFFLE,
    blocksize=0
)

_STATIC_ARRAY_COMPRESSOR_CONFIG = dict(
    id='blosc',
    **_STATIC_ARRAY_COMPRESSOR_PARAMS
)

_STATIC_ARRAY_COMPRESSOR = Blosc(**_STATIC_ARRAY_COMPRESSOR_PARAMS)


def _dict_to_bytes(d: Dict) -> bytes:
    return _str_to_bytes(json.dumps(d, indent=2))


def _bytes_to_dict(b: bytes) -> Dict:
    return json.loads(_bytes_to_str(b))


def _str_to_bytes(s: str):
    return bytes(s, encoding='utf-8')


def _bytes_to_str(b: bytes) -> str:
    return b.decode('utf-8')


class RemoteStore(MutableMapping, ABC):

    def __init__(self,
                 dataset_id: str,
                 cube_params: Mapping[str, Any] = None,
                 observer: Callable = None,
                 trace_store_calls=False):
        if not cube_params:
            cube_params = {}
        # self._dimensions = self.get_dimensions()

    @abstractmethod
    def get_encoding(self, band_name: str) -> Dict[str, Any]:
        """
        Get the encoding settings for band (variable) *var_name*.
        Must at least contain "dtype" whose value is a numpy array-protocol type string.
        Refer to https://docs.scipy.org/doc/numpy/reference/arrays.interface.html#arrays-interface
        and zarr format 2 spec.
        """

    @abstractmethod
    def get_attrs(self, var_name: str) -> Dict[str, Any]:
        """
        Get any metadata attributes for band (variable) *var_name*.
        """

    @abstractmethod
    def get_dimensions(self) -> Mapping[str, int]:
        pass

    @abstractmethod
    def get_coords_data(self, dataset_id: str) -> dict:
        pass

    @abstractmethod
    def get_variable_data(self, dataset_id: str,
                          variable_names: Dict[str, int]):
        pass

    @abstractmethod
    def get_time_ranges(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        pass

    def request_bbox(self, x_tile_index: int, y_tile_index: int) \
            -> Tuple[float, float, float, float]:
        x_tile_size, y_tile_size = self.cube_config.tile_size

        x_index = x_tile_index * x_tile_size
        y_index = y_tile_index * y_tile_size

        x01, _, _, y02 = self.cube_config.bbox
        spatial_res = self.cube_config.spatial_res

        x1 = x01 + spatial_res * x_index
        x2 = x01 + spatial_res * (x_index + x_tile_size)
        y1 = y02 - spatial_res * (y_index + y_tile_size)
        y2 = y02 - spatial_res * y_index

        return x1, y1, x2, y2

    def request_time_range(self, time_index: int) \
            -> Tuple[pd.Timestamp, pd.Timestamp]:
        start_time, end_time = self._time_ranges[time_index]
        if self.cube_config.time_tolerance:
            start_time -= self.cube_config.time_tolerance
            end_time += self.cube_config.time_tolerance
        return start_time, end_time

    def _add_static_array(self, name: str, array: np.ndarray, attrs: Dict):
        shape = list(map(int, array.shape))
        dtype = str(array.dtype.str)
        order = "C"
        array_metadata = {
            "zarr_format": 2,
            "chunks": shape,
            "shape": shape,
            "dtype": dtype,
            "fill_value": None,
            "compressor": _STATIC_ARRAY_COMPRESSOR_CONFIG,
            "filters": None,
            "order": order,
        }
        chunk_key = '.'.join(['0'] * array.ndim)
        self._vfs[name] = _str_to_bytes('')
        self._vfs[name + '/.zarray'] = _dict_to_bytes(array_metadata)
        self._vfs[name + '/.zattrs'] = _dict_to_bytes(attrs)
        self._vfs[name + '/' + chunk_key] = \
            _STATIC_ARRAY_COMPRESSOR.encode(array.tobytes(order=order))

    def _add_remote_array(self,
                          name: str,
                          shape: List[int],
                          chunks: List[int],
                          encoding: Dict[str, Any],
                          attrs: Dict):
        array_metadata = dict(zarr_format=2,
                              shape=shape,
                              chunks=chunks,
                              compressor=None,
                              fill_value=None,
                              filters=None,
                              order='C')
        array_metadata.update(encoding)
        self._vfs[name] = _str_to_bytes('')
        self._vfs[name + '/.zarray'] = _dict_to_bytes(array_metadata)
        self._vfs[name + '/.zattrs'] = _dict_to_bytes(attrs)
        nums = np.array(shape) // np.array(chunks)
        indexes = itertools.product(*tuple(map(range, map(int, nums))))
        for index in indexes:
            filename = '.'.join(map(str, index))
            # noinspection PyTypeChecker
            self._vfs[name + '/' + filename] = name, index

    def _fetch_chunk(self,
                     key: str,
                     var_name: str,
                     chunk_index: Tuple[int, ...]) -> bytes:
        if len(chunk_index) == 4:
            time_index, y_chunk_index, x_chunk_index, band_index = chunk_index
        else:
            time_index, y_chunk_index, x_chunk_index = chunk_index

        # TODO: check with norman about the exact functionality of both the
        #  below functions, replace with my functions in cmems class
        request_bbox = self.request_bbox(x_chunk_index, y_chunk_index)
        request_time_range = self.request_time_range(time_index)

        t0 = time.perf_counter()
        try:
            exception = None
            chunk_data = self.fetch_chunk(key,
                                          var_name,
                                          chunk_index,
                                          bbox=request_bbox,
                                          time_range=request_time_range)
        except Exception as e:
            exception = e
            chunk_data = None
        duration = time.perf_counter() - t0

        for observer in self._observers:
            observer(band_name=var_name,
                     chunk_index=chunk_index,
                     bbox=request_bbox,
                     time_range=request_time_range,
                     duration=duration,
                     exception=exception)

        if exception:
            raise exception

        return chunk_data

    @abstractmethod
    def fetch_chunk(self,
                    key: str,
                    var_name: str,
                    chunk_index: Tuple[int, ...],
                    bbox: Tuple[float, float, float, float],
                    time_range: Tuple[pd.Timestamp, pd.Timestamp]) -> bytes:
        """
        Fetch chunk data from remote.

        param key: The original chunk key being retrieved.
        param var_name: Variable name.
        param chunk_index: 3D chunk index (time, y, x).
        param bbox: Requested bounding box in coordinate units of the CRS.
        param time_range: Requested time range.
        return: chunk data as raw bytes.
        """
        pass

    @property
    def _class_name(self):
        return self.__module__ + '.' + self.__class__.__name__

    ##########################################################################
    # Zarr Store (MutableMapping) implementation
    ##########################################################################

    def keys(self) -> KeysView[str]:
        if self._trace_store_calls:
            print(f'{self._class_name}.keys()')
        return self._vfs.keys()

    def listdir(self, key: str) -> Iterable[str]:
        if self._trace_store_calls:
            print(f'{self._class_name}.listdir(key={key!r})')
        if key == '':
            return list((k for k in self._vfs.keys() if '/' not in k))
        else:
            prefix = key + '/'
            start = len(prefix)
            return list((k for k in self._vfs.keys()
                         if k.startswith(prefix) and k.find('/', start) == -1))

    def getsize(self, key: str) -> int:
        if self._trace_store_calls:
            print(f'{self._class_name}.getsize(key={key!r})')
        return len(self._vfs[key])

    def __iter__(self) -> Iterator[str]:
        if self._trace_store_calls:
            print(f'{self._class_name}.__iter__()')
        return iter(self._vfs.keys())

    def __len__(self) -> int:
        if self._trace_store_calls:
            print(f'{self._class_name}.__len__()')
        return len(self._vfs.keys())

    def __contains__(self, key) -> bool:
        if self._trace_store_calls:
            print(f'{self._class_name}.__contains__(key={key!r})')
        return key in self._vfs

    def __getitem__(self, key: str) -> bytes:
        if self._trace_store_calls:
            print(f'{self._class_name}.__getitem__(key={key!r})')
        value = self._vfs[key]
        if isinstance(value, tuple):
            return self._fetch_chunk(key, *value)
        return value

    def __setitem__(self, key: str, value: bytes) -> None:
        if self._trace_store_calls:
            print(
                f'{self._class_name}.__setitem__(key={key!r}, value={value!r})'
            )
        raise TypeError(f'{self._class_name} is read-only')

    def __delitem__(self, key: str) -> None:
        if self._trace_store_calls:
            print(f'{self._class_name}.__delitem__(key={key!r})')
        raise TypeError(f'{self._class_name} is read-only')


class CmemsChunkStore(RemoteStore):
    _SAMPLE_TYPE_TO_DTYPE = {
        'UINT8': '|u1',
        'UINT16': '<u2',
        'UINT32': '<u4',
        'INT8': '|u1',
        'INT16': '<u2',
        'INT32': '<u4',
        'FLOAT32': '<f4',
        'FLOAT64': '<f8',
    }

    def __init__(self,
                 cmems: cmems,
                 dataset_id: str,
                 cube_params: Mapping[str, Any] = None,
                 observer: Callable = None,
                 trace_store_calls=False):
        self.cmems = cmems
        self.metadata = self.cmems.consolidate_metadata()
        # TODO: Think about this to get all attributes of a dataset_id
        self._attrs = {}
        self._observers = [observer] if observer is not None else []
        self._trace_store_calls = trace_store_calls

        super().__init__(dataset_id,
                         cube_params,
                         observer=observer,
                         trace_store_calls=trace_store_calls)

    def fetch_chunk(self,
                    key: str,
                    var_name: str,
                    chunk_index: Tuple[int, ...],
                    bbox: Tuple[float, float, float, float],
                    time_range: Tuple[pd.Timestamp, pd.Timestamp]) -> bytes:
        start_time, end_time = time_range
        iso_start_date = start_time.tz_localize(None).isoformat()
        iso_end_date = end_time.tz_localize(None).isoformat()
        dim_indexes = self._get_dimension_indexes_for_chunk(var_name,
                                                            chunk_index)
        request = {}
        data = self.cmems.get_data_chunk(request, dim_indexes)
        if not data:
            raise KeyError(f'{key}: cannot fetch chunk for variable '
                           f'{var_name!r} and time_range {time_range!r}.')
        return data

    def _get_dimension_indexes_for_chunk(self, var_name: str,
                                         chunk_index: Tuple[int, ...]) -> tuple:
        dim_indexes = []
        var_dimensions = self.get_attrs(var_name).get('file_dimensions', [])
        chunk_sizes = self.get_attrs(var_name).get('file_chunk_sizes', [])
        offset = 0
        # dealing with the case that time has been added as additional first
        # dimension
        if len(chunk_index) > len(chunk_sizes):
            offset = 1
        for i, var_dimension in enumerate(var_dimensions):
            if var_dimension == 'time':
                dim_indexes.append(slice(None, None, None))
                continue
            # TODO: The below dim_size doesn't exist yet
            dim_size = self._dimensions.get(var_dimension, -1)
            if dim_size < 0:
                raise ValueError(
                    f'Could not determine size of dimension {var_dimension}')
            # TODO: Below data_offset doesn't exist yet
            data_offset = self._dimension_chunk_offsets.get(var_dimension, 0)
            start = data_offset + chunk_index[i + offset] * chunk_sizes[i]
            end = min(start + chunk_sizes[i], data_offset + dim_size)
            dim_indexes.append(slice(start, end))
        return tuple(dim_indexes)

    def get_encoding(self, var_name: str) -> Dict[str, Any]:
        encoding_dict = {
            'fill_value': self.get_attrs(var_name).get('fill_value'),
            'dtype': self.get_attrs(var_name).get('data_type')}
        return encoding_dict

    def get_attrs(self, var_name: str) -> Dict[str, Any]:
        if var_name not in self._attrs:
            self._attrs[var_name] = copy.deepcopy(
                self.cmems.metadata.get('var_info', {}).get(var_name, {}))
        return self._attrs[var_name]

    def get_time_ranges(self, cube_params: Mapping[str, Any] = None) \
            -> List[Tuple]:
        # TODO: check what exactly is the function for, whether it should return
        #  the time ranges provided in cube_config or of a dataset? Also see if
        #  it needs to be overriden in child class
        # time_start, time_end = cube_params.get("time_range")
        return self.cmems.get_time_ranges_from_dataset()

    def get_dimensions(self) -> Mapping[str, int]:
        # TODO: Modify the below logic, can't get dimemsions without var name
        # dimensions = {}
        # variables = self.cmems.metadata['var_info']
        # for var in variables:
        #         dimensions.append('time')
        return self.cmems.metadata['var_info']['dimensions']

    def get_coords_data(self, dataset_id: str) -> dict:
        pass

    def get_variable_data(self, dataset_id: str,
                          variable_dict: Dict[str, int]):
        return self.cmems.get_variable_data(dataset_id,
                                            variable_dict)
