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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import asyncio
from functools import cache
from typing import List, Dict, Any, Optional
import os
import logging

import nest_asyncio
from urllib.parse import urlsplit
from pydap.cas.get_cookies import setup_session
from owslib.fes import SortBy
from owslib.fes import SortProperty
from owslib.csw import CatalogueServiceWeb

from .constants import CAS_URL
from .constants import ODAP_SERVER
from .constants import DATABASE
from .constants import CSW_URL
from .constants import TOTAL_RECORDS_CSW
from .constants import STEP_SIZE

_LOG = logging.getLogger('xcube')


class Cmems:
    """
        Represents the CMEMS opendap API
        :param cmems_username: CMEMS UserID
        :param cmems_password: CMEMS User Password
        :param cas_url: CMEMS cas url
        :param csw_url: CMEMS csw url
        :param databases: databases available - nrt (near real time)
        or my(multi-year)
        :param server: odap server

    """

    def __init__(self,
                 cmems_username: Optional[str] = None,
                 cmems_password: Optional[str] = None,
                 cas_url: str = CAS_URL,
                 csw_url: str = CSW_URL,
                 databases: List = DATABASE,
                 server: str = ODAP_SERVER):
        self.cmems_username = cmems_username if cmems_username is not None \
            else os.getenv('CMEMS_USERNAME')
        self.cmems_password = cmems_password if cmems_password is not None \
            else os.getenv('CMEMS_PASSWORD')
        self.valid_opendap_url = None
        self._csw_url = csw_url
        self.databases = databases
        self.odap_server = server
        self.metadata = {}
        self.opendap_dataset_ids = {}

        if not self.cmems_username or not self.cmems_password:
            raise ValueError('CmemsDataStore needs cmems credentials in '
                             'environment variables CMEMS_USERNAME and '
                             'CMEMS_PASSWORD or to be provided as '
                             'store params cmems_username and '
                             'cmems_password')

        self.session = setup_session(cas_url, self.cmems_username,
                                     self.cmems_password)

        cast_gc = self.session.cookies.get_dict().get('CASTGC')
        if cast_gc:
            # required by Central Authentication Service (CAS).
            # The setup_session function from pydap.cas.get_cookies is used to
            # establish a session with the CAS
            self.session.cookies.set("CASTGC", cast_gc)

    def get_opendap_urls(self, data_id) -> List[str]:
        """
        Constructs opendap urls given the dataset id
        :return: List of opendap urls
        """
        urls = []
        for i in range(len(self.databases)):
            urls.append(os.path.join("https://" + self.databases[i] + "." +
                                     self.odap_server + data_id))

        return urls

    async def get_record_from_csw(self, csw, rec) -> None:
        """
        Construct opendap dataset_ids from csw records, maxrecords is the
        predefined number of records fetched from csw.getrecords2
        """
        sortby = SortBy([SortProperty("dc:title", "ASC")])
        csw.getrecords2(
            startposition=rec + 1,
            maxrecords=50,
            sortby=sortby,
            esn='full')
        for record in csw.records.values():
            if len(record.uris) > 0:
                for uris in record.uris:
                    if uris['protocol'] == 'WWW:OPENDAP':
                        if uris['url']:
                            opendap_uri = uris['url']
                            scheme, netloc, path, query, fragment = \
                                urlsplit(opendap_uri)
                            split_paths = path.split('/')
                            self.opendap_dataset_ids[split_paths[-1]] = \
                                record.title

    async def get_csw_records_concurrently(self, csw) -> None:
        """
        get csw records concurrently
        """
        tasks = []
        # records returned by csw for CMEMS is 281 currently, since it is not
        # possible know this before calling csw.getrecords, TOTAL_RECORDS_CSW
        # and STEP_SIZE is declared as constants
        for rec in range(0, TOTAL_RECORDS_CSW, STEP_SIZE):
            task = asyncio.ensure_future(self.get_record_from_csw(csw, rec))
            tasks.append(task)
        await asyncio.gather(*tasks)

    @cache
    def get_all_dataset_ids(self) -> Dict[str, Any]:
        """
        get all the opendap dataset ids by iterating through all CSW records
        currently by using asyncio
        :return: Dictionary of opendap dataset ids
        """
        csw = CatalogueServiceWeb(self._csw_url, timeout=60)
        # Workaround for RuntimeError: event loop is already running for JNB
        nest_asyncio.apply()
        asyncio.run(self.get_csw_records_concurrently(csw))
        return self.opendap_dataset_ids

    def dataset_names(self) -> List[str]:
        if self.opendap_dataset_ids:
            return self.opendap_dataset_ids.keys()
        else:
            return self.get_all_dataset_ids().keys()
