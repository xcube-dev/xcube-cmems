import copy
import logging
import os
import warnings
from datetime import time
from random import random
from typing import List, Dict, Tuple, Optional, Union, Mapping
from urllib.parse import urlsplit, urlunsplit, quote

import aiohttp
import numpy as np
from distributed.compatibility import WINDOWS
from pydap.handlers.dap import BaseProxy
from pydap.handlers.dap import SequenceProxy
from pydap.handlers.dap import unpack_data
from pydap.lib import BytesReader
from pydap.lib import combine_slices
from pydap.lib import fix_slice
from pydap.lib import hyperslab
from pydap.lib import walk
from pydap.model import BaseType, SequenceType, GridType
from pydap.parsers import parse_ce
from pydap.parsers.dds import build_dataset
from pydap.parsers.das import parse_das, add_attributes
from six.moves.urllib.parse import urlsplit, urlunsplit
from pydap.cas.get_cookies import setup_session

from .constants import CAS_URL
from .constants import ODAP_SERVER
from .constants import DATABASE

_LOG = logging.getLogger('xcube')

_SAMPLE_TYPE_TO_DTYPE = {
        'uint8': '|u1',
        'uint16': '<u2',
        'uint32': '<u4',
        'int8': '|u1',
        'int16': '<u2',
        'int32': '<u4',
        'float32': '<f4',
        'float64': '<f8',
    }

class Cmems:
    """
        Represents the CMEMS  Data Portal

        :param opendap_url: The base URL to the opendap dataset
    """

    def __init__(self,
                 cmems_user: str,
                 cmems_user_password: str,
                 dataset_id: str,
                 cas_url: str = CAS_URL,
                 databases: List = DATABASE,
                 server: str = ODAP_SERVER
                 ):
        self._cas_url = cas_url
        self._user = cmems_user
        self._password = cmems_user_password
        self.dataset_id = dataset_id
        self.databases = databases
        self.odap_server = server

    def _get_opendap_urls(self):
        urls = []
        for i in range(len(self.databases)):
            urls.append(os.path.join("https://" + self.databases[1] + "." +
                                     self.odap_server + self.dataset_id))

        return urls

    def get_dataset_metadata(self):
        session = self._create_session(self._user, self._password)
        urls = self._get_opendap_urls()
        return self._get_metadata(urls[0], session) #TODO: Handle for both urls

    def _get_metadata(self, opendap_url, session):
        if opendap_url == 'None':
            _LOG.warning(f'Dataset is not accessible via Opendap')
            return {}, {}
        dataset = self._get_opendap_dataset(session, opendap_url)
        if not dataset:
            _LOG.warning(f'Could not extract information about variables '
                         f'and attributes from {opendap_url}')
            return {}, {}
        variable_infos = {}
        for key in dataset.keys():
            fixed_key = key.replace('%2E', '_').replace('.', '_')
            data_type = dataset[key].dtype.name
            var_attrs = copy.deepcopy(dataset[key].attributes)
            var_attrs['orig_data_type'] = data_type
            if '_FillValue' in var_attrs:
                var_attrs['fill_value'] = var_attrs['_FillValue']
                var_attrs.pop('_FillValue')
            else:
                if data_type in _SAMPLE_TYPE_TO_DTYPE:
                    data_type = _SAMPLE_TYPE_TO_DTYPE[data_type]
                    var_attrs['fill_value'] = \
                        self._determine_fill_value(np.dtype(data_type))
                else:
                    warnings.warn(f'Variable "{fixed_key}" has no fill value, '
                                  f'cannot set one. For parts where no data is '
                                  f'available you will see random values. This '
                                  f'is usually the case when data is missing '
                                  f'for a time step.',
                                  )
            var_attrs['size'] = dataset[key].size
            var_attrs['shape'] = list(dataset[key].shape)
            if len(var_attrs['shape']) == 0:
                var_attrs['shape'] = [var_attrs['size']]
            if '_ChunkSizes' in var_attrs and 'DODS' not in var_attrs:
                var_attrs['chunk_sizes'] = var_attrs['_ChunkSizes']
                var_attrs.pop('_ChunkSizes')
            else:
                var_attrs['chunk_sizes'] = var_attrs['shape']
            # do this to ensure that chunk size is never bigger than shape
            if isinstance(var_attrs['chunk_sizes'], List):
                for i, chunksize in enumerate(var_attrs['chunk_sizes']):
                    var_attrs['chunk_sizes'][i] = min(chunksize,
                                                      var_attrs['shape'][i])
            else:
                var_attrs['chunk_sizes'] = min(var_attrs['chunk_sizes'],
                                               var_attrs['shape'][0])
            if type(var_attrs['chunk_sizes']) == int:
                var_attrs['file_chunk_sizes'] = var_attrs['chunk_sizes']
            else:
                var_attrs['file_chunk_sizes'] = \
                    copy.deepcopy(var_attrs['chunk_sizes'])
            var_attrs['data_type'] = data_type
            var_attrs['dimensions'] = list(dataset[key].dimensions)
            var_attrs['file_dimensions'] = \
                copy.deepcopy(var_attrs['dimensions'])
            variable_infos[fixed_key] = var_attrs

        return variable_infos, dataset.attributes

    def _determine_fill_value(self, dtype):
        if np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).max
        if np.issubdtype(dtype, np.inexact):
            return np.nan

    def _create_session(self, username, password):
        session = setup_session(self._cas_url, username, password)
        session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
        return session

    def get_opendap_dataset(self, url: str):
        session = self._create_session(self._user, self._password)
        return self._get_opendap_dataset(session, url)

    def _get_result_dict(self, session, url: str):
        res_dict = {}
        self._get_content_from_opendap_url(url, 'dds', res_dict, session)
        self._get_content_from_opendap_url(url, 'das', res_dict, session)
        if 'das' in res_dict:
            res_dict['das'] = res_dict['das'].replace(
                '        Float32 valid_min -Infinity;\n', '')
            res_dict['das'] = res_dict['das'].replace(
                '        Float32 valid_max Infinity;\n', '')
        # result_dicts[url] = res_dict
        return res_dict

    def _get_opendap_dataset(self, session, url: str):
        res_dict = self._get_result_dict(session, url)
        if 'dds' not in res_dict or 'das' not in res_dict:
            print('Could not open opendap url. No dds or das file provided.')
            return
        if res_dict['dds'] == '':
            print('Could not open opendap url. dds file is empty.')
            return
        dataset = build_dataset(res_dict['dds'])
        add_attributes(dataset, parse_das(res_dict['das']))

        # remove any projection from the url, leaving selections
        scheme, netloc, path, query, fragment = urlsplit(url)
        projection, selection = parse_ce(query)
        url = urlunsplit((scheme, netloc, path, '&'.join(selection), fragment))

        # now add data proxies
        for var in walk(dataset, BaseType):
            var.data = BaseProxy(url, var.id, var.dtype, var.shape)
        for var in walk(dataset, SequenceType):
            template = copy.copy(var)
            var.data = SequenceProxy(url, template)

        # apply projections
        for var in projection:
            target = dataset
            while var:
                token, index = var.pop(0)
                target = target[token]
                if isinstance(target, BaseType):
                    target.data.slice = fix_slice(index, target.shape)
                elif isinstance(target, GridType):
                    index = fix_slice(index, target.array.shape)
                    target.array.data.slice = index
                    for s, child in zip(index, target.maps):
                        target[child].data.slice = (s,)
                elif isinstance(target, SequenceType):
                    target.data.slice = index

        # retrieve only main variable for grid types:
        for var in walk(dataset, GridType):
            var.set_output_grid(True)

        return dataset

    def _get_content_from_opendap_url(self, url: str, part: str,
                                      res_dict: dict, session):
        scheme, netloc, path, query, fragment = urlsplit(url)
        url = urlunsplit((scheme, netloc, path + f'.{part}', query, fragment))
        resp = self._get_response(session, url)
        if resp:
            res_dict[part] = resp.content
            res_dict[part] = str(res_dict[part], 'utf-8')

    def _get_data_from_opendap_dataset(self, dataset, session,
                                       variable_name, slices):
        proxy = dataset[variable_name].data
        if type(proxy) == list:
            proxy = proxy[0]
        # build download url
        index = combine_slices(proxy.slice, fix_slice(slices, proxy.shape))
        scheme, netloc, path, query, fragment = urlsplit(proxy.baseurl)
        url = urlunsplit((
            scheme, netloc, path + '.dods',
            quote(proxy.id) + hyperslab(index) + '&' + query,
            fragment)).rstrip('&')
        # download and unpack data
        resp = self.get_response(session, url)
        if not resp:
            _LOG.warning(f'Could not read response from "{url}"')
            return None
        content = resp.content
        dds, data = content.split(b'\nData:\n', 1)
        dds = str(dds, 'utf-8')
        # Parse received dataset:
        dataset = build_dataset(dds)
        try:
            dataset.data = unpack_data(BytesReader(data), dataset)
        except ValueError:
            _LOG.warning(f'Could not read data from "{url}"')
            return None
        return dataset[proxy.id].data

    def _get_response(self, session: aiohttp.ClientSession, url: str) -> \
            Optional[aiohttp.ClientResponse]:
        resp = session.request(method='GET', url=url)
        if resp.status_code == 200:
            return resp
        else:
            return None

    def get_variable_data(self,
                          variable_dict: Dict[str, int],
                          start_time: str = '1900-01-01T00:00:00',
                          end_time: str = '3001-12-31T00:00:00'):
        dimension_data = (self._get_var_data, variable_dict, start_time,
                          end_time)
        return dimension_data

    def _get_var_data(self,
                      session,
                      variable_dict: Dict[str, int],
                      start_time: str,
                      end_time: str):
        request = dict(startDate=start_time,
                       endDate=end_time
                       )
        opendap_urls = self._get_opendap_urls(session, request)
        var_data = {}
        if not opendap_urls:
            return var_data

        try:
            dataset = self._get_opendap_dataset(session, opendap_urls[0])
        except:
            dataset = self._get_opendap_dataset(session, opendap_urls[1])

        if not dataset:
            return var_data
        for var_name in variable_dict:
            if var_name in dataset:
                var_data[var_name] = dict(size=dataset[var_name].size,
                                          shape=dataset[var_name].shape,
                                          chunkSize=dataset[var_name].
                                          attributes.get('_ChunkSizes'))
                if dataset[var_name].size < 512 * 512:
                    data = self._get_data_from_opendap_dataset(
                        dataset,
                        session,
                        var_name,
                        (slice(None, None, None),))
                    if data is None:
                        var_data[var_name]['data'] = []
                    else:
                        var_data[var_name]['data'] = data
                else:
                    var_data[var_name]['data'] = []
            else:
                var_data[var_name] = dict(
                    size=variable_dict[var_name],
                    chunkSize=variable_dict[var_name],
                    data=list(range(variable_dict[var_name])))
        return var_data

    def get_data_chunk(self, session, request: Dict, dim_indexes: Tuple) -> \
            Optional[bytes]:
        var_name = request['varNames'][0]
        opendap_urls = self._get_opendap_urls()
        try:
            dataset = self._get_opendap_dataset(session, opendap_urls[0])
        except:
            dataset = self._get_opendap_dataset(session, opendap_urls[1])
        if not dataset:
            return None
        # data_type = self._data_sources[request['drsId']].get('variable_infos', {}) \
        #     .get(var_name, {}).get('data_type')
        data = self._get_data_from_opendap_dataset(dataset, self,
                                                   var_name, dim_indexes)
        if data is None:
            return None
        data = np.array(data, copy=False, dtype=np.float32)
        return data.flatten().tobytes()
