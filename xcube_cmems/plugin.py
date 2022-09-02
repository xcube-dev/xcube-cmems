# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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
from xcube.util import extension
from xcube.constants import EXTENSION_POINT_DATA_OPENERS
from xcube.constants import EXTENSION_POINT_DATA_STORES
from xcube_cmems.constants import DATASET_OPENER_ID
from xcube_cmems.constants import DATA_STORE_ID


def init_plugin(ext_registry: extension.ExtensionRegistry):
    ext_registry.add_extension(
        loader=extension.import_component(
            'xcube_cmems.store:CmemsDataStore'),
        point=EXTENSION_POINT_DATA_STORES,
        name=DATA_STORE_ID,
        description='Cmems DataStore',
        data_store_notices=[dict(id='terminologyClarification',
                                 title='Terminology Clarification',
                                 # content=data_store_notice_content,
                                 intent='primary',
                                 icon='info-sign'),
                            dict(id='dataCompleteness',
                                 title='Data Completeness',
                                 # content=data_completeness_content,
                                 intent='warning',
                                 icon='warning-sign')])
    ext_registry.add_extension(
        loader=extension.import_component(
            'xcube_cmems.store:CmemsDatasetOpener'),
        point=EXTENSION_POINT_DATA_OPENERS,
        name=DATASET_OPENER_ID,
        description='xarray.Dataset in Zarr format'
                    ' from CMEMS API'
    )
