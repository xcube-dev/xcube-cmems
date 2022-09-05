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

import os
import logging
from urllib.parse import urlsplit
from pydap.cas.get_cookies import setup_session
from typing import List
from typing import Dict
from typing import Any
from .constants import CAS_URL
from .constants import ODAP_SERVER
from .constants import DATABASE
from .constants import CSW_URL
from owslib.fes import SortBy
from owslib.fes import SortProperty
from owslib.csw import CatalogueServiceWeb

_LOG = logging.getLogger('xcube')


class Cmems:
    """
        Represents the CMEMS opendap API
        :param cmems_user: CMEMS UserID
        :param cmems_user_password: CMEMS User Password
        :param dataset_id: opendap dataset id
        :param cas_url: CMEMS cas url
        :param csw_url: CMEMS csw url
        :param databases: databases available - nrt (near real time)
        or my(multi-year)
        :param server: odap server

    """

    def __init__(self,
                 cmems_user: str,
                 cmems_user_password: str,
                 dataset_id: str,
                 cas_url: str = CAS_URL,
                 csw_url: str = CSW_URL,
                 databases: List = DATABASE,
                 server: str = ODAP_SERVER
                 ):
        self.valid_opendap_url = None
        self._csw_url = csw_url
        self.dataset_id = dataset_id
        self.databases = databases
        self.odap_server = server
        self.metadata = {}
        self.opendap_dataset_ids = {}

        self.session = setup_session(cas_url, cmems_user,
                                     cmems_user_password)
        self.session.cookies.set("CASTGC",
                                 self.session.cookies.get_dict()['CASTGC']
                                 )

    def get_opendap_urls(self) -> List[str]:
        """
        Constructs opendap urls given the dataset id
        :return: List of opendap urls
        """
        urls = []
        for i in range(len(self.databases)):
            urls.append(os.path.join("https://" + self.databases[i] + "." +
                                     self.odap_server + self.dataset_id))

        return urls

    @staticmethod
    def get_csw_records(csw, pagesize=10, max_records=300) -> Dict[Any, Any]:
        """
        Iterate max_records/pagesize times until the requested value in
        max_records is reached.
        return: CSW records
        """
        # Iterate over sorted results.
        sortby = SortBy([SortProperty("dc:title", "ASC")])
        csw_records = {}
        start_position = 0
        next_record = getattr(csw, "results", 1)
        while next_record != 0:
            csw.getrecords2(
                startposition=start_position,
                maxrecords=pagesize,
                sortby=sortby,
                esn='full'
            )
            csw_records.update(csw.records)
            if csw.results["nextrecord"] == 0:
                break
            start_position += pagesize + 1
            if start_position >= max_records:
                break
        csw.records.update(csw_records)
        return csw_records

    def get_all_dataset_ids(self) -> Dict[str, Any]:
        """
        get all the opendap dataset ids by iterating through all CSW records
        :return: Dictionary of opendap dataset ids
        """
        csw = CatalogueServiceWeb(self._csw_url, timeout=60)
        csw_rec = self.get_csw_records(csw, pagesize=10, max_records=2000)
        csw_obj_list = list(csw_rec.values())
        for record in csw_obj_list:
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
        return self.opendap_dataset_ids

    def dataset_names(self) -> List[str]:
        return self.get_all_dataset_ids().keys()
