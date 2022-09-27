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

import unittest
import os

from dotenv import load_dotenv
from mock import patch

from xcube_cmems.cmems import Cmems

from .sample_data import get_all_dataset_results


class CmemsTest(unittest.TestCase):

    @classmethod
    def _create_cmems_instance(cls):
        cmems = Cmems()
        return cmems

    def test_get_opendap_urls(self):
        cmems = self._create_cmems_instance()
        dataset_id = "dataset-bal-analysis-forecast-wav-hourly"
        urls = cmems.get_opendap_urls(dataset_id)
        self.assertEqual('https://nrt.cmems-du.eu/thredds/dodsC/dataset-bal'
                         '-analysis-forecast-wav-hourly', urls[0])
        self.assertEqual('https://my.cmems-du.eu/thredds/dodsC/dataset-bal'
                         '-analysis-forecast-wav-hourly', urls[1])

    @patch.object(Cmems, "get_all_dataset_ids")
    def test_get_data_ids(self, mock_get_all_dataset_ids):
        mock_get_all_dataset_ids.return_value = get_all_dataset_results()
        cmems = self._create_cmems_instance()
        self.assertEqual(520, len(cmems.get_all_dataset_ids()))

    @patch.object(Cmems, "dataset_names")
    def test_get_dataset_names(self, mock_dataset_names):
        cmems = self._create_cmems_instance()
        dataset_dict = get_all_dataset_results()
        mock_dataset_names.return_value = dataset_dict.keys()
        self.assertEqual(520, len(cmems.dataset_names()))
