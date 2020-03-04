#!/usr/bin/env python3
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
# with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
# and limitations under the License.

import mxnet as mx
import numpy as np

filename = 'glove.6B.50d.npz'
np_loaded = np.load(filename, allow_pickle=True)

idx_to_vec = mx.nd.array(np_loaded['idx_to_vec'])
mx.ndarray.save('idx_to_vec.mx', idx_to_vec)

unknown_token = str(np_loaded['unknown_token'])
with open('unknown_token.txt', 'w') as f:
    f.write(unknown_token)

idx_to_token = [str(s) for s in np_loaded['idx_to_token']]
with open('idx_to_token.txt', 'w') as f:
    for s in idx_to_token:
        print(idx_to_token, file=f)
