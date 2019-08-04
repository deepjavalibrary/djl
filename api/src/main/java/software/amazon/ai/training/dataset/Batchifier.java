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
package software.amazon.ai.training.dataset;

import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;

public interface Batchifier {

    Batchifier STACK_BATCHIFIER = NDArrays::stack;
    Batchifier LIST_BATCHIFIER = NDArrays::concat;

    default NDList batch(NDList[] inputs) {
        NDList list = new NDList(inputs.length);
        for (NDList input : inputs) {
            list.add(batch(input));
        }

        return list;
    }

    default NDArray batch(NDArray[] arrays) {
        return batch(new NDList(arrays));
    }

    NDArray batch(NDList list);
}
