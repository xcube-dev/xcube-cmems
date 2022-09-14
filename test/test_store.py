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
import numpy as np
import xarray as xr
import xcube.core.store.descriptor as xcube_des
from dotenv import load_dotenv
from xcube_cmems.cmems import Cmems
from xcube_cmems.store import CmemsDatasetOpener
from xcube_cmems.store import CmemsDataOpener
from xcube_cmems.store import CmemsDataStore
from xcube.util.jsonschema import JsonObjectSchema
from mock import patch
from .sample_data import create_cmems_dataset
from .sample_data import get_all_dataset_results


class CmemsDataOpenerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset_id = "dataset-bal-analysis-forecast-wav-hourly"
        load_dotenv()
        cmems_user = os.getenv("CMEMS_USER")
        cmems_user_password = os.getenv("CMEMS_PASSWORD")
        cmems_params = {'cmems_user': cmems_user,
                        'cmems_user_password': cmems_user_password,
                        'dataset_id': self.dataset_id
                        }
        self.opener = CmemsDatasetOpener(**cmems_params)

    @patch.object(CmemsDataOpener, "open_dataset")
    def test_open_data(self, mock_open_dataset):
        mock_open_dataset.return_value = create_cmems_dataset()
        mocked_ds = self.opener.open_data(self.dataset_id)
        self.assertIsInstance(mocked_ds, xr.Dataset)

    def test_describe_data(self):
        data_des = self.opener.describe_data(self.dataset_id)
        self.assertIsInstance(data_des, xcube_des.DatasetDescriptor)
        self.assertEqual(('2018-12-01', '2022-09-20'), data_des.time_range)
        self.assertEqual('dataset-bal-analysis-forecast-wav-hourly',
                         data_des.data_id)
        self.assertEqual(('time', 'lat', 'lon'),
                         data_des.data_vars.get('VHM0').dims)
        self.assertEqual(764, data_des.dims['lon'])
        self.assertEqual(775, data_des.dims['lat'])
        self.assertEqual(33336, data_des.dims['time'])
        self.assertTrue('VTPK' in data_des.data_vars)
        self.assertEqual(3, data_des.data_vars['VTPK'].ndim)
        self.assertEqual(('time', 'lat', 'lon'),
                         data_des.data_vars['VTPK'].dims)
        self.assertEqual(np.float32,
                         data_des.data_vars['VTPK'].dtype)
        self.assertEqual('epsg:4326', data_des.crs)
        self.assertEqual('0D', data_des.time_period)


class CmemsDataStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset_id = "dataset-bal-analysis-forecast-wav-hourly"
        load_dotenv()
        cmems_user = os.getenv("CMEMS_USER")
        cmems_user_password = os.getenv("CMEMS_PASSWORD")
        cmems_params = {'cmems_user': cmems_user,
                        'cmems_user_password': cmems_user_password,
                        'dataset_id': self.dataset_id
                        }
        self.datastore = CmemsDataStore(**cmems_params)

    @patch.object(CmemsDataStore, "get_data_ids")
    def test_get_all_data_ids(self, mock_get_data_ids):
        mock_get_data_ids.return_value = get_all_dataset_results()
        dataset_ids = self.datastore.get_data_ids()
        dataset_ids = list(dataset_ids)
        self.assertEqual(520, len(dataset_ids))

    def test_describe_data(self):
        data_des = self.datastore.describe_data(self.dataset_id)
        self.assertIsInstance(data_des, xcube_des.DatasetDescriptor)
        self.assertEqual(('2018-12-01', '2022-09-20'), data_des.time_range)
        self.assertEqual('dataset-bal-analysis-forecast-wav-hourly',
                         data_des.data_id)
        self.assertEqual(('time', 'lat', 'lon'),
                         data_des.data_vars.get('VHM0').dims)
        self.assertEqual(764, data_des.dims['lon'])
        self.assertEqual(775, data_des.dims['lat'])
        self.assertEqual(33336, data_des.dims['time'])
        self.assertTrue('VTPK' in data_des.data_vars)
        self.assertEqual(3, data_des.data_vars['VTPK'].ndim)
        self.assertEqual(('time', 'lat', 'lon'),
                         data_des.data_vars['VTPK'].dims)
        self.assertEqual(np.float32,
                         data_des.data_vars['VTPK'].dtype)
        self.assertEqual('epsg:4326', data_des.crs)
        self.assertEqual('0D', data_des.time_period)

    @patch.object(CmemsDataOpener, "open_dataset")
    def test_open_data(self, mock_open_dataset):
        mock_open_dataset.return_value = create_cmems_dataset()
        mocked_ds = self.datastore.open_data(self.dataset_id)
        self.assertIsInstance(mocked_ds, xr.Dataset)

    def test_get_open_data_params(self):
        open_params = self.datastore.get_open_data_params_schema(
            self.dataset_id)
        self.assertIsInstance(open_params, JsonObjectSchema)

    def test_get_data_types(self):
        self.assertEqual(('dataset',), self.datastore.get_data_types())

    @patch.object(Cmems, "dataset_names")
    def test_has_data(self, mock_dataset_names):
        dataset_dict = get_all_dataset_results()
        mock_dataset_names.return_value = dataset_dict.keys()
        self.assertEqual(True, self.datastore.has_data(self.dataset_id))
