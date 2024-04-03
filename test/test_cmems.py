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
import pathlib
import os
import unittest

from xcube_cmems.cmems import Cmems
from unittest.mock import patch, MagicMock


class CmemsTest(unittest.TestCase):

    # the mocks the click.confirm function, process to confirm actions like
    # overwriting files.
    @patch("click.confirm", return_value=True)
    def setUp(self, mock_confirm):
        # Setup environment variables for testing
        os.environ["CMEMS_USERNAME"] = "testuser"
        os.environ["CMEMS_PASSWORD"] = "testpass"
        self.configuration_file_directory = pathlib.Path.cwd()

    @patch("click.confirm", return_value=True)
    @patch("xcube_cmems.cmems.cm.describe")
    def test_get_datasets_with_titles(self, mock_describe, mock_confirm):
        # Mock the response from cm.describe
        mock_describe.return_value = {
            "products": [
                {
                    "title": "Product A",
                    "datasets": [
                        {"dataset_id": "dataset1"},
                        {"dataset_id": "dataset2"},
                    ],
                },
                {"title": "Product B", "datasets": [{"dataset_id": "dataset3"}]},
            ]
        }
        cmems = Cmems()
        datasets_info = cmems.get_datasets_with_titles()

        # Expected result based on the mocked describe response
        expected_result = [
            {"title": "Product A", "dataset_id": "dataset1"},
            {"title": "Product A", "dataset_id": "dataset2"},
            {"title": "Product B", "dataset_id": "dataset3"},
        ]

        self.assertEqual(datasets_info, expected_result)

    @patch("click.confirm", return_value=True)
    @patch("xcube_cmems.cmems.cm.open_dataset")
    def test_open_dataset(self, mock_open_dataset, mock_confirm):
        # Mock the response from cm.open_dataset
        mock_dataset = MagicMock()
        mock_open_dataset.return_value = mock_dataset
        cmems_instance = Cmems(
            configuration_file_directory=self.configuration_file_directory
        )

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
