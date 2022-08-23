import unittest
import pandas as pd
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

    def test_get_metadata(self):
        cmems = self._create_cmems_instance\
            ("dataset-bal-analysis-forecast-wav-hourly")
        var_info, dataset_attr = cmems.get_valid_opendap_metadata()
        self.assertIsNotNone(var_info)
        self.assertIsNotNone(dataset_attr)
        self.assertEqual(20, len(var_info))
        print(var_info)
        print(dataset_attr)

    def test_get_uuid_from_json(self):
        cmems = self._create_cmems_instance \
            ("dataset-bal-analysis-forecast-wav-hourly")
        uuid = cmems.get_csw_uuid_from_did()
        self.assertEqual('3183791b-0f94-4697-8fe8-3c7db19a624c',
                         uuid)

    def test_set_metadata_from_csw(self):
        cmems = self._create_cmems_instance \
            ("dataset-bal-analysis-forecast-wav-hourly")
        cmems.set_metadata_from_csw("3183791b-0f94-4697-8fe8-3c7db19a624c")
        self.assertEqual((9.0, 53.0, 30.0, 66.0),
                         cmems.metadata['bbox'])
        self.assertEqual('urn:ogc:def:crs:EPSG:6.6:4326',
                         cmems.metadata['crs'])

    def test_get_data_chunk(self):
        cmems = self._create_cmems_instance \
            ("dataset-bal-analysis-forecast-wav-hourly")
        request = dict(varNames=['VHM0'],
                       startDate='2020-06-16T00:00:00',
                       endDate='2020-06-17T00:00:00')
        dim_indexes = (
            slice(None, None, None), slice(0, 18, 40), slice(0, 120, 40)
            # slice(None, None), slice(0, 179), slice(0, 359)
        )
        data_bytes = cmems.get_data_chunk(request, dim_indexes)
        self.assertEqual(393264, len(data_bytes))
        print(len(data_bytes))

    def test_start_and_end_time(self):
        cmems = self._create_cmems_instance("med-cmcc-sal-int-m")
        start_time, end_time = cmems.get_start_and_end_time()
        self.assertEqual(pd.Timestamp('2020-06-16 00:00:00'), start_time)
        self.assertEqual(pd.Timestamp('2021-11-16 00:00:00'), end_time)

    def test_consolidate_metadata(self):
        cmems = self._create_cmems_instance \
            ("dataset-bal-analysis-forecast-wav-hourly")
        cmems.consolidate_metadata()
        self.assertEqual(4, len(cmems.metadata.keys()))
        self.assertEqual(['var_info', 'dataset_attr', 'bbox', 'crs'],
                         list(cmems.metadata.keys()))
        print(cmems.metadata)
