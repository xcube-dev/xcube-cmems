import copy
from abc import ABC
import itertools
import json
# import logging
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
from xcube_cmems.constants import COMMON_COORD_VAR_NAMES
from xcube_cmems.constants import LOG

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
    """
    A remote Zarr Store.

    :param data_id: The identifier of the data resource
    :param cube_params: A mapping containing additional parameters to define
        the data set.
    :param observer: An optional callback function called when remote requests
        are mode: observer(**kwargs).
    :param trace_store_calls: Whether store calls shall be printed
        (for debugging).
    """

    def __init__(self,
                 data_id: str,
                 cube_params: Mapping[str, Any] = None,
                 observer: Callable = None,
                 trace_store_calls=False):
        if not cube_params:
            cube_params = {}

        self._observers = [observer] if observer is not None else []
        self._trace_store_calls = trace_store_calls

        self._variable_names = cube_params.get('variable_names',
                                               self.get_all_variable_names())

        self._time_ranges = self.get_time_ranges(data_id, cube_params)

        LOG.debug('Determined time ranges')
        if not self._time_ranges:
            raise ValueError('Could not determine any valid time stamps')

        t_array = [s.to_pydatetime()
                   + 0.5 * (e.to_pydatetime() - s.to_pydatetime())
                   for s, e in self._time_ranges]
        t_array = np.array(t_array).astype('datetime64[s]').astype(np.int64)
        t_bnds_array = \
            np.array(self._time_ranges).astype('datetime64[s]').astype(np.int64)
        time_coverage_start = self._time_ranges[0][0]
        time_coverage_end = self._time_ranges[-1][1]

        bbox = cube_params.get('bbox', None)
        lon_size = -1
        lat_size = -1

        self._dimensions = self.get_dimensions()
        self._num_data_var_chunks_not_in_vfs = 0
        coords_data = self.get_coords_data(data_id)
        LOG.debug('Determined coordinates')
        coords_data['time'] = {}
        coords_data['time']['size'] = len(t_array)
        coords_data['time']['data'] = t_array
        if 'time_bounds' in coords_data:
            coords_data.pop('time_bounds')
        coords_data['time_bnds'] = {}
        coords_data['time_bnds']['size'] = len(t_bnds_array)
        coords_data['time_bnds']['data'] = t_bnds_array
        sorted_coords_names = list(coords_data.keys())
        sorted_coords_names.sort()

        for coord_name in sorted_coords_names:
            if coord_name == 'time' or coord_name == 'time_bnds':
                continue
            # @TODO TMH If a bounding box has been passed in the cube params,
            # adjust lat/lon values to only hold those values

            coord_data = coords_data[coord_name]['data']
            coord_attrs = self.get_attrs(coord_name)
            coord_attrs['_ARRAY_DIMENSIONS'] = coord_attrs['dimensions']

            # @TODO TMH If a bounding box has been passed in the cube params,

            if len(coord_data) > 0:
                coord_array = np.array(coord_data)
                self._add_static_array(coord_name, coord_array, coord_attrs)
            else:
                shape = list(coords_data[coord_name].
                             get('shape', coords_data[coord_name].get('size')))
                chunk_size = coords_data[coord_name]['chunkSize']
                if not isinstance(chunk_size, List):
                    chunk_size = [chunk_size]
                encoding = self.get_encoding(coord_name)
                self._add_remote_array(coord_name, shape, chunk_size,
                                       encoding, coord_attrs)

        time_attrs = {
            "_ARRAY_DIMENSIONS": ['time'],
            "units": "seconds since 1970-01-01T00:00:00Z",
            "calendar": "proleptic_gregorian",
            "standard_name": "time",
            "bounds": "time_bnds",
        }
        time_bnds_attrs = {
            "_ARRAY_DIMENSIONS": ['time', 'bnds'],
            "units": "seconds since 1970-01-01T00:00:00Z",
            "calendar": "proleptic_gregorian",
            "standard_name": "time_bnds",
        }

        self._add_static_array('time', t_array, time_attrs)
        self._add_static_array('time_bnds', t_bnds_array, time_bnds_attrs)

        coordinate_names = [coord for coord in coords_data.keys()
                            if coord not in COMMON_COORD_VAR_NAMES]
        coordinate_names = ' '.join(coordinate_names)

        global_attrs = dict(
            Conventions='CF-1.7',
            coordinates=coordinate_names,
            title=data_id,
            date_created=pd.Timestamp.now().isoformat(),
            time_coverage_start=time_coverage_start.isoformat(),
            time_coverage_end=time_coverage_end.isoformat(),
            time_coverage_duration=
                (time_coverage_end - time_coverage_start).isoformat(),
        )

        # setup Virtual File System (vfs)
        self._vfs = {
            '.zgroup': _dict_to_bytes(dict(zarr_format=2)),
            '.zattrs': _dict_to_bytes(global_attrs)
        }

        for variable_name in self._variable_names:
            var_encoding = self.get_encoding(variable_name)
            var_attrs = self.get_attrs(variable_name)
            dimensions = var_attrs.get('dimensions', [])
            var_attrs.update(_ARRAY_DIMENSIONS=dimensions)
            chunk_sizes = var_attrs.get('chunk_sizes', [-1] * len(dimensions))
            sizes = []
            for i, coord_name in enumerate(dimensions):
                if coord_name in coords_data:
                    sizes.append(coords_data[coord_name]['size'])
                else:
                    sizes.append(self._dimensions.get(coord_name))
                if chunk_sizes[i] == -1:
                    chunk_sizes[i] = sizes[i]
            var_attrs['shape'] = sizes
            var_attrs['size'] = math.prod(sizes)
            var_attrs['chunk_sizes'] = chunk_sizes
            self._add_remote_array(variable_name,
                                   sizes,
                                   chunk_sizes,
                                   var_encoding,
                                   var_attrs)
        LOG.debug(f"Added a total of {len(self._variable_names)} variables "
                  f"to the data set")
        cube_params['variable_names'] = self._variable_names
        global_attrs['history'] = [dict(
            program=f'{self._class_name}',
            cube_params=cube_params
        )]
        self._consolidate_metadata()

    @abstractmethod
    def get_encoding(self, band_name: str) -> Dict[str, Any]:
        """
        Get the encoding settings for band (variable) *band_name*.
        Must at least contain "dtype" whose value is a numpy array-protocol type string.
        Refer to https://docs.scipy.org/doc/numpy/reference/arrays.interface.html#arrays-interface
        and zarr format 2 spec.
        """

    @abstractmethod
    def get_time_ranges(self, data_id: str, cube_params: Mapping[str, Any]) \
            -> List[Tuple]:
        """
        Retrieve a list of time ranges. A time range consists of a start and end time.
        There is one time range per time step.
        It cube_params contains a field 'time_range', only time steps within this range
        will be considered.

        :param data_id: The id of the dataset for which time steps shall be retrieved
        :param cube_params: A mapping which might hold additional parameters
        :return: A list of tuples consisting of a one start and one end time
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> Mapping[str, int]:
        """
        :return: a mapping of dimension names to sizes
        """

    @abstractmethod
    def get_coords_data(self, dataset_id: str) -> dict:
        """
        Returns a mapping from a coordinate name to its size, shape, chunksize
        and, in case it is not too large, a data array holding the coordinate data.

        :param dataset_id:  The id of the dataset
        :return:the mapping described above
        """

    @abstractmethod
    def get_attrs(self, var_name: str) -> Dict[str, Any]:
        """
        Get any metadata attributes for variable *variable_name*.
        """

    @abstractmethod
    def get_all_variable_names(self) -> List[str]:
        """
        Returns a list of all the names of the variables in this data set
        :return:
        """

    @abstractmethod
    def get_variable_data(self, dataset_id: str,
                          variable_names: Dict[str, int]):
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
                     band_name: str,
                     chunk_index: Tuple[int, ...]) -> bytes:
        if len(chunk_index) == 4:
            time_index, y_chunk_index, x_chunk_index, band_index = chunk_index
        else:
            time_index, y_chunk_index, x_chunk_index = chunk_index

        request_bbox = self.request_bbox(x_chunk_index, y_chunk_index)
        request_time_range = self.request_time_range(time_index)

        t0 = time.perf_counter()
        try:
            exception = None
            chunk_data = self.fetch_chunk(key,
                                          band_name,
                                          chunk_index,
                                          bbox=request_bbox,
                                          time_range=request_time_range)
        except Exception as e:
            exception = e
            chunk_data = None
        duration = time.perf_counter() - t0

        for observer in self._observers:
            observer(band_name=band_name,
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

        :param key: The original chunk key being retrieved.
        :param var_name: Variable name.
        :param chunk_index: 3D chunk index (time, y, x).
        :param bbox: Requested bounding box in coordinate units of the CRS.
        :param time_range: Requested time range.
        :return: chunk data as raw bytes.
        """
        pass

    def _consolidate_metadata(self):
        # Consolidate metadata to suppress warning:  (#69)
        #
        # RuntimeWarning: Failed to open Zarr store with consolidated
        # metadata, falling back to try reading non-consolidated
        # metadata. ...
        #
        metadata = dict()
        for k, v in self._vfs.items():
            if k == '.zattrs' or k.endswith('/.zattrs') \
                    or k == '.zarray' or k.endswith('/.zarray') \
                    or k == '.zgroup' or k.endswith('/.zgroup'):
                metadata[k] = _bytes_to_dict(v)
        self._vfs['.zmetadata'] = _dict_to_bytes(
            dict(zarr_consolidated_format=1, metadata=metadata)
        )

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
        self._cmems = cmems
        self._metadata = self._cmems.get_dataset_metadata(dataset_id)
        super().__init__(dataset_id,
                         cube_params,
                         observer=observer,
                         trace_store_calls=trace_store_calls)

    def get_time_ranges(self, data_id: str, cube_params: Mapping[str, Any]) \
            -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        # @TODO THM Check whether this method is applicable like this
        time_start, time_end = cube_params.get(
            'time_range',
            (pd.Timestamp('1970-01-01'), pd.Timestamp.now())
        )
        # @TODO THM Set time period according for data. If time is irregular, the code
        # below cannot be used
        time_period = None
        time_ranges = []
        time_now = time_start
        while time_now <= time_end:
            time_next = time_now + time_period
            time_ranges.append((time_now, time_next))
            time_now = time_next
        return time_ranges

    def get_dimensions(self) -> Mapping[str, int]:
        # @TODO THM Implement
        pass

    def get_coords_data(self, dataset_id: str) -> dict:
        # @TODO THM Implement
        pass

    def get_attrs(self, var_name: str) -> Dict[str, Any]:
        # @TODO THM Implement
        pass

    def get_all_variable_names(self) -> List[str]:
        # @TODO THM Implement
        pass

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
        data = self._cmems.get_data_chunk(request, dim_indexes)

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
            dim_size = self._dimensions.get(var_dimension, -1)
            if dim_size < 0:
                raise ValueError(
                    f'Could not determine size of dimension {var_dimension}')
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
                self._metadata.get('variable_infos', {}).get(var_name, {}))
        return self._attrs[var_name]

    def get_dimensions(self) -> Mapping[str, int]:
        return copy.copy(self._metadata['dimensions'])

    def get_coords_data(self, dataset_id: str) -> dict:
        pass

    def get_variable_data(self, dataset_id: str,
                          variable_dict: Dict[str, int]):
        return self._cmems.get_variable_data(dataset_id,
                                             variable_dict)
                                             # self._time_ranges[0][0].strftime(
                                             #     _TIMESTAMP_FORMAT),
                                             # self._time_ranges[0][1].strftime(
                                             #     _TIMESTAMP_FORMAT))
