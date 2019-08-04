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

import java.io.IOException;
import java.util.Iterator;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.util.Pair;

public interface Dataset {

    /**
     * Create an iterator for the specified {@code usage}, with random iteration order.
     *
     * @param usage The dataset (train or test)
     * @param batchSize batch size for the iterator
     * @return an iterator for the specified {@code usage}
     * @throws IOException if IO error occurs
     */
    default Iterator<Pair<NDList, NDList>> getData(Usage usage, int batchSize) throws IOException {
        return getData(usage, batchSize, true);
    }

    /**
     * Create an iterator for the specified {@code usage}, with random iteration order.
     *
     * @param usage the dataset (train or test)
     * @param batchSize batch size for the iterator
     * @param shuffle shuffle the dataset
     * @return an iterator for the specified {@code usage}
     * @throws IOException if IO error occurs
     */
    default Iterator<Pair<NDList, NDList>> getData(Usage usage, int batchSize, boolean shuffle)
            throws IOException {
        Sampler sampler;
        if (shuffle) {
            sampler = new RandomSampler();
        } else {
            sampler = new SequenceSampler();
        }
        return getData(usage, batchSize, sampler);
    }

    /**
     * Create an iterator for the specified {@code usage}, with specified {@link Sampler}.
     *
     * @param usage the dataset (train or test)
     * @param batchSize batch size for the iterator
     * @param sampler a {@link Sampler} to selecting results
     * @return an iterator for the specified {@code usage}
     * @throws IOException if IO error occurs
     */
    Iterator<Pair<NDList, NDList>> getData(Usage usage, int batchSize, Sampler sampler)
            throws IOException;

    enum Usage {
        TRAIN,
        TEST,
        VALIDATION
    }
}
