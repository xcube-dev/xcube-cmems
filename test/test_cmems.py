# The MIT License (MIT)
# Copyright (c) 2022-2024 by the xcube development team and contributors
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
from types import SimpleNamespace

from xcube_cmems.cmems import Cmems
from unittest.mock import patch, MagicMock


class CmemsTest(unittest.TestCase):

    def setUp(self):
        # Setup environment variables for testing
        os.environ["COPERNICUS_MARINE_SERVICE_USERNAME"] = "testuser"
        os.environ["COPERNICUS_MARINE_SERVICE_PASSWORD"] = "testpass"

    @patch("xcube_cmems.cmems.cm.describe")
    def test_get_datasets_with_titles(self, mock_describe):
        # Fake datasets
        dataset1 = SimpleNamespace(dataset_id="dataset1", dataset_name="Dataset 1")
        dataset2 = SimpleNamespace(dataset_id="dataset2", dataset_name="Dataset 2")
        dataset3 = SimpleNamespace(dataset_id="dataset3", dataset_name="Dataset 3")

        # Fake products
        product_a = SimpleNamespace(title="Product A", datasets=[dataset1, dataset2])
        product_b = SimpleNamespace(title="Product B", datasets=[dataset3])

        # Fake catalogue
        mock_catalogue = SimpleNamespace(products=[product_a, product_b])
        mock_describe.return_value = mock_catalogue

        cmems = Cmems()
        datasets_info = cmems.get_datasets_with_titles()

        expected = [
            {"dataset_id": "dataset1", "title": "Product A - Dataset 1"},
            {"dataset_id": "dataset2", "title": "Product A - Dataset 2"},
            {"dataset_id": "dataset3", "title": "Product B - Dataset 3"},
        ]
        self.assertEqual(datasets_info, expected)

    @patch("xcube_cmems.cmems.cm.open_dataset")
    def test_open_dataset(self, mock_open_dataset):
        # Mock the response from cm.open_dataset
        mock_dataset = MagicMock()
        mock_open_dataset.return_value = mock_dataset
        cmems_instance = Cmems()
        result = cmems_instance.open_dataset("dataset1")
        self.assertEqual(result, mock_dataset)

        # Testing with a non-existing dataset
        mock_open_dataset.side_effect = KeyError("Dataset not found")
        result = cmems_instance.open_dataset("non_existing_dataset")
        self.assertIsNone(result)

    @patch("click.confirm", return_value=True)
    def test_open_data_for_not_exsiting_dataset(self, mock_confirm):
        cmems = Cmems()
        self.assertIsNone(
            cmems.open_dataset("dataset-bal-analysis-forecast" "-wav-hourly"),
            "Expected the method to return None for a " "non-existing dataset",
        )
