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
package ai.djl.training;

import ai.djl.ndarray.NDList;
import ai.djl.training.dataset.Batch;

/**
 * {@code DataManager} is an interface that is used primarily in the {@link Trainer} to manipulate
 * data from the dataset to be appropriate for the current model.
 */
public class DataManager {
    public static final DataManager DEFAULT_DATA_MANAGER = new DataManager() {};

    /**
     * Fetches data from the given {@link Batch} in required form.
     *
     * @param batch the {@link Batch} to fetch data from
     * @return an {@link NDList} with the appropriate data
     */
    public NDList getData(Batch batch) {
        return batch.getData();
    }

    /**
     * Fetches labels from the given {@link Batch} in required form.
     *
     * @param batch the {@link Batch} to fetch labels from
     * @return an {@link NDList} with the appropriate labels
     */
    public NDList getLabels(Batch batch) {
        return batch.getLabels();
    }
}
