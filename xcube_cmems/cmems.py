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
from typing import List, Optional

import copernicusmarine as cm
import numpy as np
import xarray as xr
from copernicusmarine import CopernicusMarineCatalogue


class Cmems:
    def __init__(
        self, cmems_username: Optional[str] = None, cmems_password: Optional[str] = None
    ):
        self.cmems_username = (
            cmems_username
            if cmems_username is not None
            else os.getenv("COPERNICUSMARINE_SERVICE_USERNAME")
        )
        self.cmems_password = (
            cmems_password
            if cmems_password is not None
            else os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD")
        )

        if not self.cmems_username or not self.cmems_password:
            raise ValueError(
                "CmemsDataStore needs cmems credentials to "
                "be provided either as "
                "environment variables COPERNICUSMARINE_SERVICE_USERNAME and "
                "COPERNICUSMARINE_SERVICE_PASSWORD, or to be "
                "provided as store params cmems_username and "
                "cmems_password"
            )

    @classmethod
    def get_datasets_with_titles(cls) -> List[dict]:
        catalogue: CopernicusMarineCatalogue = cm.describe(disable_progress_bar=True)
        datasets_info: List[dict] = []
        for product in catalogue.products:
            product_title = product.title
            for dataset in product.datasets:
                datasets_info.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "title": f"{product_title} - {dataset.dataset_name}",
                    }
                )
        return datasets_info

    def to_json_serializable(self, obj):
        """Convert NumPy types and nested structures to JSON-serializable types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): self.to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.to_json_serializable(i) for i in obj]
        return obj

    def sanitize_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """Sanitize dataset and variable attributes for JSON serialization."""
        for var in ds.data_vars:
            ds[var].attrs = {
                str(k): self.to_json_serializable(v) for k, v in ds[var].attrs.items()
            }
        ds.attrs = {str(k): self.to_json_serializable(v) for k, v in ds.attrs.items()}
        return ds

    def open_dataset(self, dataset_id, **open_params) -> xr.Dataset:
        try:

            ds = cm.open_dataset(
                dataset_id=dataset_id,
                username=self.cmems_username,
                password=self.cmems_password,
                **open_params,
            )
            ds = self.sanitize_attrs(ds)
            return ds
        except KeyError as e:
            print(f"Error: {e}.")
            print(
                f"The dataset '{dataset_id}' was not found in the Copernicus "
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
