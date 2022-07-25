import copy
import warnings
from datetime import time, datetime
from random import random
from typing import Optional, Dict, Tuple
from urllib.parse import urlsplit, urlunsplit, quote

import logging
import aiohttp as aiohttp
import numpy as np
from pydap.handlers.dap import BaseProxy, SequenceProxy, unpack_data
from pydap.lib import walk, fix_slice, hyperslab, BytesReader
from pydap.model import BaseType, SequenceType, GridType
from pydap.parsers import parse_ce
from pydap.parsers.das import add_attributes, parse_das
from pydap.parsers.dds import build_dataset
from pydap.cas.get_cookies import setup_session
from pydap.lib import combine_slices

import os


_LOG = logging.getLogger('xcube')
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"

def get_opendap_dataset(session, url: str):
    res_dict = get_result_dict(session, url)
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


def get_result_dict(session, url: str):
    res_dict = {}
    get_content_from_opendap_url(url, 'dds', res_dict, session)
    get_content_from_opendap_url(url, 'das', res_dict, session)
    if 'das' in res_dict:
        res_dict['das'] = res_dict['das'].replace(
            '        Float32 valid_min -Infinity;\n', '')
        res_dict['das'] = res_dict['das'].replace(
            '        Float32 valid_max Infinity;\n', '')
    # result_dicts[url] = res_dict
    return res_dict


def get_content_from_opendap_url(url: str, part: str, res_dict: dict, session):
    scheme, netloc, path, query, fragment = urlsplit(url)
    url = urlunsplit((scheme, netloc, path + f'.{part}', query, fragment))
    resp = get_response(session, url)
    if resp:
        res_dict[part] = resp.content
        res_dict[part] = str(res_dict[part], 'utf-8')


def get_response(session: aiohttp.ClientSession, url: str) -> \
        Optional[aiohttp.ClientResponse]:
    resp = session.request(method='GET', url=url)
    if resp.status_code == 200:
        return resp
    else:
        return None


def get_data_from_opendap_dataset(dataset, session, variable_name,
                                  slices):
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
    resp = get_response(session, url)
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


def get_opendap_url(request):
    # url = "https://my.cmems-du.eu/thredds/dodsC/"
    # dataset_id = ""
    # os.path.join(url + dataset_id + request)

    return "https://nrt.cmems-du.eu/thredds/dodsC/" \
           "dataset-bal-analysis-forecast-wav-hourly?VHM0[0:1:0][0:1:0][0:1:0]"

def get_feature_list(session, request):
    start_date_str = request['startDate']
    start_date = datetime.strptime(start_date_str, _TIMESTAMP_FORMAT)
    end_date_str = request['endDate']
    end_date = datetime.strptime(end_date_str, _TIMESTAMP_FORMAT)
    feature_list = []

def get_data_chunk(session, request: Dict, dim_indexes: Tuple) -> \
Optional[bytes]:
    var_name = request['varNames'][0]
    opendap_url = get_opendap_url(request)
    if not opendap_url:
        return None
    dataset = get_opendap_dataset(session, opendap_url)
    if not dataset:
        return None
    # await self._ensure_all_info_in_data_sources(session,
    #                                             [request.get('drsId')])
    # data_type = self._data_sources[request['drsId']].get('variable_infos', {}) \
    #     .get(var_name, {}).get('data_type')
    data = get_data_from_opendap_dataset(dataset, session,
                                         var_name, dim_indexes)
    if data is None:
        return None
    data = np.array(data, copy=False, dtype=np.float32)
    return data.flatten().tobytes()


cas_url = 'https://cmems-cas.cls.fr/cas/login'
USERNAME = "tmorbagal"
PASSWORD = "Tejas@1993"
session = setup_session(cas_url, USERNAME, PASSWORD)
session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
dataset_url = "https://my.cmems-du.eu/thredds/dodsC/bs-ulg-car-int-m"
dataset = get_opendap_dataset(session, dataset_url)
print(dataset)
print(dataset.keys())

request = dict(varNames=['VHM0'])
dim_indexes = (slice(None, None), slice(0, 179), slice(0, 359))
data = get_data_chunk(session, request, dim_indexes)