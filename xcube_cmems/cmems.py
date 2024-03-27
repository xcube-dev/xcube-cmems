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
import pathlib
from typing import List, Optional

import copernicusmarine as cm
import xarray as xr


class Cmems:

    def __init__(
        self,
        cmems_username: Optional[str] = None,
        cmems_password: Optional[str] = None,
        configuration_file_directory=pathlib.Path.cwd(),
    ):
        self.cmems_username = (
            cmems_username
            if cmems_username is not None
            else os.getenv("CMEMS_USERNAME")
        )
        self.cmems_password = (
            cmems_password
            if cmems_password is not None
            else os.getenv("CMEMS_PASSWORD")
        )

        if not self.cmems_username or not self.cmems_password:
            raise ValueError(
                "CmemsDataStore needs cmems credentials to "
                "be provided either as "
                "environment variables CMEMS_USERNAME and "
                "CMEMS_PASSWORD, or to be "
                "provided as store params cmems_username and "
                "cmems_password"
            )
        cm.login(
            username=self.cmems_username,
            password=self.cmems_password,
            configuration_file_directory=configuration_file_directory,
        )

    @classmethod
    def get_datasets_with_titles(cls) -> List[dict]:
        catalogue: dict = cm.describe(include_datasets=True)
        datasets_info: List[dict] = []
        for product in catalogue["products"]:
            product_title = product["title"]
            for dataset in product["datasets"]:
                dataset_id: str = dataset["dataset_id"]
                datasets_info.append({"title": product_title, "dataset_id": dataset_id})
        return datasets_info

    def open_dataset(self, data_id) -> xr.Dataset:
        try:
            ds = cm.open_dataset(
                dataset_id=data_id,
                username=self.cmems_username,
                password=self.cmems_password,
            )
            return ds
        except KeyError as e:
            print(f"Error: {e}.")
            print(
                f"The dataset '{data_id}' was not found in the Copernicus "
                f"Marine catalogue."
            )
            print(
                "Please check the dataset ID for typos or use the "
                "'get_dataset_ids' command to list the available "
                "datasets."
            )
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
