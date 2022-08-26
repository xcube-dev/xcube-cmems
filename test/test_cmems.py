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

from xcube_cmems.cmems import Cmems
from dotenv import load_dotenv


class CmemsTest(unittest.TestCase):

    @classmethod
    def _create_cmems_instance(cls, dataset_id):
        # load the environment variables
        load_dotenv()
        cmems = Cmems(os.getenv("CMEMS_USER"), os.getenv("CMEMS_PASSWORD"),
                      dataset_id)
        return cmems

    def test_get_opendap_urls(self):
        dataset_id = "dataset-bal-analysis-forecast-wav-hourly"
        cmems = self._create_cmems_instance(dataset_id)
        self.assertEqual(['https://my.cmems-du.eu/thredds/dodsC/dataset-bal'
                          '-analysis-forecast-wav-hourly',
                          'https://nrt.cmems-du.eu/thredds/dodsC/dataset-bal'
                          '-analysis-forecast-wav-hourly'],
                         cmems.get_opendap_urls()
                         )

    def test_get_data_ids(self):
        dataset_id = "dataset-bal-analysis-forecast-wav-hourly"
        cmems = self._create_cmems_instance(dataset_id)
        data_ids = cmems.get_all_dataset_ids()
        self.assertEqual(520, len(data_ids))
