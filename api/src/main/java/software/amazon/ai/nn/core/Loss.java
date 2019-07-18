/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package software.amazon.ai.nn.core;

import software.amazon.ai.Block;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.util.PairList;

public interface Loss extends Block {

    NDArray forward(NDArray pred, NDArray label);

    @Override
    default NDList forward(NDList inputs, PairList<String, String> params) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException(
                    "Required two NDArray input, found " + inputs.size());
        }
        return new NDList(forward(inputs.get(0), inputs.get(1)));
    }
}
