#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import io
import numpy as np


def to_npz(np_list: list) -> bytearray:
    """
    Converts list of numpy array to bytes.
    :param np_list: The list of numpy array to convert.
    :return: The result bytes.
    """
    buf = io.BytesIO()
    np.savez(buf, *np_list)
    buf.seek(0)
    return buf.read(-1)


def from_npz(data: bytearray) -> list:
    """
    Converts bytes to list of numpy array.
    :param data: The bytes to convert.
    :return: The result list of numpy array.
    """
    result = []
    npz = np.load(io.BytesIO(data))
    for item in npz.items():
        result.append(item[1])
    return result
