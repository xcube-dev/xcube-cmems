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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import pandas as pd
from typing import List

UNIT_MAPPER = dict(
    minutes='m',
    milliseconds='ms',
    nanoseconds='ns',
    seconds='s',
)


def get_timestamp(time_value: float, units: str) -> pd.Timestamp:
    if 'since' not in units:
        return pd.Timestamp(time_value)
    unit, offset_stamp = units.split('since')
    offset = pd.Timestamp(offset_stamp.strip())
    delta = pd.Timedelta(value=time_value, unit=UNIT_MAPPER[unit.strip()])
    return offset + delta


def get_timestamps(time_values: List[float], units: str) -> List[pd.Timestamp]:
    if 'since' not in units:
        return [pd.Timestamp(time_value) for time_value in time_values]
    unit, offset_stamp = units.split('since')
    offset = pd.Timestamp(offset_stamp.strip())
    timestamps = [offset +
                  pd.Timedelta(value=time_value,
                               unit=UNIT_MAPPER[unit.strip()])
                  for time_value in time_values]
    return timestamps


def get_timestamp_as_iso_string(time_value: float, units: str):
    return get_timestamp(time_value, units).isoformat()


