import pandas as pd
import unittest

from xcube_cmems.utils import get_timestamp
from xcube_cmems.utils import get_timestamps


class GetTimestampTest(unittest.TestCase):

    def test_get_timestamp_no_units_set(self):
        self.assertEqual(pd.Timestamp(2000000), get_timestamp(2000000, ''))

    def test_get_timestamp_seconds_since_1970(self):
        self.assertEqual(
            pd.Timestamp('2001-09-09 01:46:40'),
            get_timestamp(1000000000, 'seconds since 1970-01-01')
        )

    def test_get_timestamp_milliseconds_since_1950(self):
        self.assertEqual(
            pd.Timestamp('1950-01-12 13:46:40'),
            get_timestamp(1000000000, 'milliseconds since 1950-01-01')
        )

    def test_get_timestamp_nanoseconds_since_1990(self):
        self.assertEqual(
            pd.Timestamp('1990-06-01 00:00:01'),
            get_timestamp(1000000000, 'nanoseconds since 1990-06-01')
        )

    def test_get_timestamp_minutes_since_1990(self):
        self.assertEqual(
            pd.Timestamp('2015-09-25 05:20:00'),
            get_timestamp(50000000, 'minutes since 1920-09-01')
        )


class GetTimestampsTest(unittest.TestCase):

    def test_get_timestamp_minutes_since_1900(self):
        time_stamps = [pd.Timestamp('2020-06-16 00:00:00'),
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
                       pd.Timestamp('2021-10-16 12:00:00')]
        time_values = [6.335424E7, 6.339816E7, 6.34428E7, 6.348672E7,
                       6.353064E7, 6.357456E7, 6.361848E7, 6.366312E7,
                       6.37056E7, 6.374808E7, 6.3792E7, 6.383592E7, 6.387984E7,
                       6.392376E7, 6.39684E7, 6.401232E7, 6.405624E7]
        self.assertEqual(
            time_stamps,
            get_timestamps(time_values, 'minutes since 1900-01-01')
        )
