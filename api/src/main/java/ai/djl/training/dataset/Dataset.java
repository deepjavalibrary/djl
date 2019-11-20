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
package ai.djl.training.dataset;

import ai.djl.ndarray.NDManager;

/** An interface to represent a dataset. Every dataset must implement this interface. */
public interface Dataset {

    /**
     * Fetches an iterator that can iterate through the {@link Dataset}.
     *
     * @param manager the dataset to iterate through
     * @return an {@link Iterable} of {@link Batch} that contains batches of data from the dataset
     */
    Iterable<Batch> getData(NDManager manager);

    /** An enum that indicates the mode - training, test or validation. */
    enum Usage {
        TRAIN,
        TEST,
        VALIDATION
    }
}
