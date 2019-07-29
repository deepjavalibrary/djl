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

import software.amazon.ai.ndarray.NDList;

public interface Dataset {

    /**
     * Create an iterator for the specified {@code usage}, with random iteration order.
     *
     * @param batchSize batch size for the iterator
     * @param usage The dataset (train or test)
     * @return an iterator for the specified {@code usage}
     */
    default Iterable<NDList> getData(Usage usage, int batchSize) {
        return getData(usage, batchSize, -1);
    }

    /**
     * Create an iterator for the specified {@code usage}, with random iteration order.
     *
     * @param batchSize batch size for the iterator
     * @param usage the dataset (train or test)
     * @param seed the random seed
     * @return an iterator for the specified {@code usage}
     */
    Iterable<NDList> getData(Usage usage, int batchSize, int seed);

    enum Usage {
        TRAIN,
        TEST,
        VALIDATION
    }
}
