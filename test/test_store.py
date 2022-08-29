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
import unittest
import types
import xarray as xr
import xcube.core.store.descriptor as xcube_des
from dotenv import load_dotenv

from xcube_cmems.store import CmemsDatasetOpener
from xcube_cmems.store import CmemsDataStore


class CmemsDataOpenerTest(unittest.TestCase):

    def setUp(self) -> None:
        dataset_id = "dataset-bal-analysis-forecast-wav-hourly"
        load_dotenv()
        cmems_user = os.getenv("CMEMS_USER")
        cmems_user_password = os.getenv("CMEMS_PASSWORD")
        cmems_params = {'cmems_user': cmems_user,
                        'cmems_user_password': cmems_user_password,
                        'dataset_id': dataset_id
                        }
        self.opener = CmemsDatasetOpener(**cmems_params)

    def test_open_data(self):
        dataset_id = "dataset-bal-analysis-forecast-wav-hourly"
        ds = self.opener.open_data(dataset_id,
                                   variable_names=['VHM0'],
                                   # bbox=[9.0, 53.0, 30.0, 66.0],
                                   time_range=['2020-06-16', '2020-07-16']
                                   )
        self.assertIsInstance(ds, xr.Dataset)

    def test_describe_data(self):
        dataset_id = "dataset-bal-analysis-forecast-wav-hourly"
        data_des = self.opener.describe_data(dataset_id)
        self.assertIsInstance(data_des,
                              xcube_des.DatasetDescriptor)


class CmemsDataStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        dataset_id = "dataset-bal-analysis-forecast-wav-hourly"
        load_dotenv()
        cmems_user = os.getenv("CMEMS_USER")
        cmems_user_password = os.getenv("CMEMS_PASSWORD")
        cmems_params = {'cmems_user': cmems_user,
                        'cmems_user_password': cmems_user_password,
                        'dataset_id': dataset_id
                        }
        self.datastore = CmemsDataStore(**cmems_params)

    def test_get_all_data_ids(self):
        dataset_ids = self.datastore.get_data_ids()
        self.assertIsInstance(dataset_ids, types.GeneratorType)
        dataset_ids = list(dataset_ids)
        self.assertEqual(520, len(dataset_ids))
