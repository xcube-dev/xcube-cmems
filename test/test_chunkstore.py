import unittest

import numpy

from xcube_cmems.chunkstore import CmemsChunkStore
from test.test_cmems import CmemsTest
import pandas as pd


class CmemsChunkstoreTest(unittest.TestCase):

    @classmethod
    def _create_chunk_store_instance(cls, dataset_id):
        cmems = CmemsTest._create_cmems_instance(dataset_id)

        chunk_store = CmemsChunkStore(cmems, dataset_id)
        return chunk_store

    # def test_get_dimensions(self):
    #     chunk_store = self._create_chunk_store_instance\
    #                   ("dataset-bal-analysis-forecast-wav-hourly")
    #     self.assertEqual(['time', 'lat', 'lon'], chunk_store.get_dimensions())

    def test_get_attributes(self):
        chunk_store = self._create_chunk_store_instance\
                      ("dataset-bal-analysis-forecast-wav-hourly")
        attrs = chunk_store.get_attrs('VHM0')
        self.assertTrue('standard_name' in attrs)
        self.assertTrue('long_name' in attrs)
        self.assertTrue('units' in attrs)
        self.assertTrue('fill_value' in attrs)
        self.assertTrue('chunk_sizes' in attrs)
        self.assertTrue('data_type' in attrs)
        self.assertTrue('dimensions' in attrs)
        self.assertEqual('float32', attrs['data_type'])
        self.assertEqual(['time', 'lat', 'lon'], attrs['dimensions'])
        print(attrs)

    def test_get_encoding(self):
        store = self._create_chunk_store_instance\
                      ("dataset-bal-analysis-forecast-wav-hourly")
        encoding_dict = store.get_encoding('VHM0')
        self.assertTrue('fill_value' in encoding_dict)
        self.assertTrue('dtype' in encoding_dict)
        self.assertFalse('compressor' in encoding_dict)
        self.assertFalse('order' in encoding_dict)
        self.assertEqual('float32', encoding_dict['dtype'])

    def test_get_time_ranges(self):
        store = self._create_chunk_store_instance("med-cmcc-sal-int-m")
        # time_range = (pd.to_datetime('2020-06-16', utc=True),
        #               pd.to_datetime('2020-07-18', utc=True))
        # cube_params = dict(time_range=time_range)
        time_ranges = store.get_time_ranges()
        self.assertEqual([pd.Timestamp('2020-06-16 00:00:00'),
                          pd.Timestamp('2020-07-16 12:00:00'),
                          pd.Timestamp('2020-08-16 12:00:00'),
                          pd.Timestamp('2020-09-16 00:00:00'),
                          pd.Timestamp('2020-10-16 12:00:00'),
                          pd.Timestamp('2020-11-16 00:00:00'),
                          pd.Timestamp('2020-12-16 12:00:00'),
                          pd.Timestamp('2021-01-16 12:00:00'),
                          pd.Timestamp('2021-02-15 00:00:00'),
                          pd.Timestamp('2021-03-16 12:00:00'),
                          pd.Timestamp('2021-04-16 00:00:00'),
                          pd.Timestamp('2021-05-16 12:00:00'),
                          pd.Timestamp('2021-06-16 00:00:00'),
                          pd.Timestamp('2021-07-16 12:00:00'),
                          pd.Timestamp('2021-08-16 12:00:00'),
                          pd.Timestamp('2021-09-16 00:00:00'),
                          pd.Timestamp('2021-10-16 12:00:00'),
                          pd.Timestamp('2021-11-16 00:00:00')], time_ranges)
        print(time_ranges)

    def test_big_dataset_get_time_ranges(self):
        # TODO: Complete this unit test (Ask Tonio when he is back)
        store = self._create_chunk_store_instance\
                ("dataset-bal-analysis-forecast-wav-hourly")
        # time_range = (pd.to_datetime('2020-06-16', utc=True),
        #               pd.to_datetime('2020-07-18', utc=True))
        # cube_params = dict(time_range=time_range)
        time_ranges = store.get_time_ranges()
        print(time_ranges)