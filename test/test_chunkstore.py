import unittest
from getpass import getpass

from xcube_cmems.chunkstore import CmemsChunkStore
from xcube_cmems.cmems import Cmems
from test.test_cmems import CmemsTest


class CmemsChunkstoreTest(unittest.TestCase):
    cmems = CmemsTest.create_cmems_instance()

    chunk_store = CmemsChunkStore(cmems,
                                 "dataset-bal-analysis-forecast-wav-hourly")
    print(chunk_store.metadata)
