import getpass
import unittest
from xcube_cmems.cmems import Cmems


class CmemsTest(unittest.TestCase):
    def test_get_metadata(self):
        PASSWORD = getpass.getpass('Enter your password: ')
        dataset_id = "dataset-bal-analysis-forecast-wav-hourly"
        cmems = Cmems("tmorbagal", PASSWORD, dataset_id)
        varible_metadata, attributes = cmems.get_dataset_metadata()
        print(varible_metadata)
        print(attributes)
