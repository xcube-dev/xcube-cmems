import getpass
import unittest
from xcube_cmems.cmems import Cmems


class CmemsTest(unittest.TestCase):

    @classmethod
    def create_cmems_instance(cls):
        PASSWORD = getpass.getpass('Enter your password: ')
        dataset_id = "dataset-bal-analysis-forecast-wav-hourly"
        cmems = Cmems("tmorbagal", PASSWORD, dataset_id)
        return cmems

    def test_get_metadata(self):
        cmems = self.create_cmems_instance()
        cmems.get_dataset_metadata()
        self.assertIsNotNone(cmems.variable_infos)
        self.assertIsNotNone(cmems.dataset_attributes)
        self.assertEqual(20, len(cmems.variable_infos))
        print(cmems.variable_infos)
        print(cmems.dataset_attributes)

    def test_get_metadata_1(self):
        cmems = self.create_cmems_instance()
        cmems.get_dataset_metadata()
        self.assertIsNotNone(cmems.variable_infos)
        self.assertIsNotNone(cmems.dataset_attributes)
        self.assertEqual(20, len(cmems.variable_infos))
        print(cmems.variable_infos)
        print(cmems.dataset_attributes)

    def test_get_uuid_from_json(self):
        cmems = self.create_cmems_instance()
        uuid = cmems.get_csw_uuid_from_did()
        self.assertEqual('0b6c9fbc-fd13-4629-b79d-88c67e1348bd',
                         uuid)

    def test_get_metadata_from_csw(self):
        cmems = self.create_cmems_instance()
        csw_record = cmems.get_metadata_from_csw("0b6c9fbc-fd13-4629-b79d"
                                                 "-88c67e1348bd")
        self.assertEqual((27.37, 40.86, 41.96, 46.8),
                         cmems.dataset_info['bbox'])
        self.assertEqual('urn:ogc:def:crs:EPSG:6.6:4326',
                         cmems.dataset_info['crs'].id)

    def test_get_data_chunk(self):
        cmems = self.create_cmems_instance()
        request = dict(varNames=['VHM0'])
        dim_indexes = (slice(None, None, None), slice(0, 40, 40), slice(0, 120, 40))
        data_bytes = cmems.get_data_chunk(request, dim_indexes)
