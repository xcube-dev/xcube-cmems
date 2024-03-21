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

import xarray as xr
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import xcube.core.store.descriptor as xcube_des
from xcube.util.jsonschema import JsonObjectSchema
from xcube_cmems.constants import DATASET_OPENER_ID
from xcube_cmems.store import CmemsDatasetOpener
from xcube_cmems.store import CmemsDataStore
from .sample_data import create_cmems_dataset


class CmemsDataOpenerTest(unittest.TestCase):

    @patch("click.confirm", return_value=True)
    def setUp(self, mock_confirm) -> None:
        with patch("click.confirm", return_value=True):
            self.dataset_id = "cmems_mod_arc_bgc_anfc_ecosmo_P1D-m"
            self.opener = CmemsDatasetOpener()

    @patch("xcube_cmems.cmems.cm.open_dataset")
    def test_subset_cube_with_open_params(self, mock_open_dataset):
        start_date = datetime.now()
        end_date = start_date + timedelta(days=7)
        dataset_id = "test_dataset"
        mock_open_dataset.return_value = xr.Dataset()
        open_params = {
            "time_range": (start_date, end_date),
            "bbox": (-10, -20, 10, 20),
            "variable_names": ["var1", "var2"],
        }
        result = self.opener.subset_cube_with_open_params(dataset_id, **open_params)
        mock_open_dataset.assert_called_once_with(
            dataset_id=dataset_id,
            start_datetime=start_date,
            end_datetime=end_date,
            minimum_longitude=-10,
            minimum_latitude=-20,
            maximum_longitude=10,
            maximum_latitude=20,
            variables=["var1", "var2"],
        )
        assert isinstance(result, xr.Dataset)

    @patch("xcube_cmems.store.CmemsDataOpener.subset_cube_with_open_params")
    @patch("xcube_cmems.store.CmemsDataOpener.get_open_data_params_schema")
    def test_open_data(self, mock_get_schema, mock_subset_cube):
        data_id = "data_id"
        mock_subset_cube.return_value = xr.Dataset()
        schema = MagicMock()
        schema.validate_instance = MagicMock()
        schema.process_kwargs_subset = MagicMock(return_value=({}, {}))
        mock_get_schema.return_value = schema

        result = self.opener.open_data(data_id)

        mock_get_schema.assert_called_once_with(data_id)
        schema.validate_instance.assert_called_once()
        schema.process_kwargs_subset.assert_called_once()
        mock_subset_cube.assert_called_once_with(data_id, **{})
        assert isinstance(result, xr.Dataset)

    @patch("xcube_cmems.cmems.cm.open_dataset")
    def test_describe_data(self, mock_open_dataset):
        mock_open_dataset.return_value = create_cmems_dataset()
        data_des = self.opener.describe_data(self.dataset_id)
        self.assertIsInstance(data_des, xcube_des.DatasetDescriptor)
        self.assertEqual(("2022-01-01", "2022-01-08"), data_des.time_range)
        self.assertEqual("cmems_mod_arc_bgc_anfc_ecosmo_P1D-m", data_des.data_id)
        self.assertEqual(("time", "lat", "lon"), data_des.data_vars.get("VHM0").dims)
        self.assertEqual(5, data_des.dims["lon"])
        self.assertEqual(5, data_des.dims["lat"])
        self.assertEqual(8, data_des.dims["time"])
        self.assertTrue("VTPK" in data_des.data_vars)
        self.assertEqual(3, data_des.data_vars["VTPK"].ndim)
        self.assertEqual(("time", "lat", "lon"), data_des.data_vars["VTPK"].dims)
        self.assertEqual("float64", data_des.data_vars["VTPK"].dtype)
        self.assertEqual("WGS 84", data_des.crs)
        self.assertEqual("1D", data_des.time_period)


class CmemsDataStoreTest(unittest.TestCase):

    @patch("click.confirm", return_value=True)
    def setUp(self, mock_confirm) -> None:
        self.dataset_id = "cmems_mod_arc_bgc_anfc_ecosmo_P1D-m"
        self.mock_datasets = [
            {"dataset_id": "id1", "title": "Title 1"},
            {"dataset_id": "id2", "title": "Title 2"},
        ]
        self.datastore = CmemsDataStore()

    @patch("xcube_cmems.cmems.Cmems.get_datasets_with_titles")
    def test_get_data_ids_without_include_attrs(self, mock_get_datasets):
        mock_get_datasets.return_value = self.mock_datasets

        expected_result = ["id1", "id2"]
        result = list(self.datastore.get_data_ids())

        self.assertEqual(result, expected_result)

    @patch("xcube_cmems.cmems.Cmems.get_datasets_with_titles")
    def test_get_data_ids_with_empty_attrs(self, mock_get_datasets):
        mock_get_datasets.return_value = self.mock_datasets

        expected_result = [("id1", {}), ("id2", {})]
        result = list(self.datastore.get_data_ids(include_attrs=[]))

        self.assertEqual(result, expected_result)

    @patch("xcube_cmems.cmems.Cmems.get_datasets_with_titles")
    def test_get_data_ids_with_title_in_attrs(self, mock_get_datasets):
        mock_get_datasets.return_value = self.mock_datasets

        expected_result = [("id1", {"title": "Title 1"}), ("id2", {"title": "Title 2"})]
        result = list(self.datastore.get_data_ids(include_attrs=["title"]))

        self.assertEqual(result, expected_result)

    @patch("xcube_cmems.cmems.cm.open_dataset")
    def test_describe_data(self, mock_open_dataset):
        mock_open_dataset.return_value = create_cmems_dataset()
        data_des = self.datastore.describe_data(self.dataset_id)
        self.assertIsInstance(data_des, xcube_des.DatasetDescriptor)
        self.assertEqual(("2022-01-01", "2022-01-08"), data_des.time_range)
        self.assertEqual("cmems_mod_arc_bgc_anfc_ecosmo_P1D-m", data_des.data_id)
        self.assertEqual(("time", "lat", "lon"), data_des.data_vars.get("VHM0").dims)
        self.assertEqual(5, data_des.dims["lon"])
        self.assertEqual(5, data_des.dims["lat"])
        self.assertEqual(8, data_des.dims["time"])
        self.assertTrue("VTPK" in data_des.data_vars)
        self.assertEqual(3, data_des.data_vars["VTPK"].ndim)
        self.assertEqual(("time", "lat", "lon"), data_des.data_vars["VTPK"].dims)
        self.assertEqual("float64", data_des.data_vars["VTPK"].dtype)
        self.assertEqual("WGS 84", data_des.crs)
        self.assertEqual("1D", data_des.time_period)

    @patch("xcube_cmems.cmems.cm.open_dataset")
    def test_open_data_with_cube_params(self, mock_open_data):
        mock_open_data.return_value = create_cmems_dataset()
        mocked_ds = self.datastore.open_data(
            self.dataset_id,
            variable_names=["VHM0"],
            time_range=("2022-01-01", "2022-01-03"),
        )
        self.assertIsInstance(mocked_ds, xr.Dataset)
        self.assertEqual(8, mocked_ds.sizes["time"])
        self.assertEqual(2, len(mocked_ds.data_vars))
        self.assertTrue("VHM0" in mocked_ds.data_vars)

    def test_get_open_data_params(self):
        open_params = self.datastore.get_open_data_params_schema(
            "cmems_obs-si_ant_phy_nrt_l3-1km_P1D"
        )
        self.assertIsInstance(open_params, JsonObjectSchema)

    def test_get_data_types(self):
        self.assertEqual(("dataset",), self.datastore.get_data_types())

    def test_get_data_opener_ids_with_valid_data_id(self):
        self.datastore.has_data = MagicMock(return_value=True)
        data_id = "valid_data_id"
        data_type = "dataset"
        expected = (DATASET_OPENER_ID,)

        result = self.datastore.get_data_opener_ids(
            data_id=data_id, data_type=data_type
        )

        assert result == expected
        self.datastore.has_data.assert_called_once_with(data_id, data_type=data_type)


class CmemsDataStoreParamsTest(unittest.TestCase):

    def test_store_for_cmems_credentials(self):
        params = {"cmems_username": "", "cmems_password": ""}
        with self.assertRaises(Exception) as e:
            CmemsDataStore(**params)
        self.assertEqual(
            "CmemsDataStore needs cmems credentials in "
            "environment variables "
            "CMEMS_USERNAME and CMEMS_PASSWORD or to "
            "be provided as store params cmems_username "
            "and cmems_password",
            f"{e.exception}",
        )
