import unittest
from getpass import getpass

from xcube_cmems.chunkstore import CmemsChunkStore
from xcube_cmems.cmems import Cmems
from test.test_cmems import CmemsTest


class CmemsChunkstoreTest(unittest.TestCase):

    @classmethod
    def _create_chunk_store_instance(cls, dataset_id):
        cmems = CmemsTest._create_cmems_instance()

        chunk_store = CmemsChunkStore(cmems, dataset_id)
        return chunk_store

    def test_get_dimensions(self):
        chunk_store = self._create_chunk_store_instance\
                      ("dataset-bal-analysis-forecast-wav-hourly")
        self.assertEqual(['time', 'lat', 'lon'], chunk_store.get_dimensions())

