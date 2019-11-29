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

import java.util.Iterator;
import java.util.List;

/**
 * An interface for sampling data items from a {@link RandomAccessDataset}.
 *
 * <p>A {@code Sampler} implementation returns an iterator of batches for the {@link
 * RandomAccessDataset}. Instead of returning the actual items, it returns the item indices.
 * Different samplers can have different ways of sampling such as sampling with or without
 * replacement.
 *
 * <p>Many of the samplers may also make use of {@link SubSampler}s which sample not in batches but
 * in individual data item indices.
 */
public interface Sampler {

    /**
     * Fetches an iterator that iterates through the given {@link RandomAccessDataset} in
     * mini-batches of indices.
     *
     * @param dataset the {@link RandomAccessDataset} to sample from
     * @return an iterator that iterates through the given {@link RandomAccessDataset} in
     *     mini-batches of indices
     */
    Iterator<List<Long>> sample(RandomAccessDataset dataset);

    /**
     * Returns the batch size of the {@code Sampler}.
     *
     * @return the batch size of the {@code Sampler}, -1 if batch size is not fixed
     */
    int getBatchSize();

    /** An interface that samples a single data item at a time. */
    interface SubSampler {

        /**
         * Fetches an iterator that iterates through the indices of the given {@link
         * RandomAccessDataset}.
         *
         * @param dataset the {@link RandomAccessDataset} to sample from
         * @return an iterator that iterates through the indices of the given {@link
         *     RandomAccessDataset}
         */
        Iterator<Long> sample(RandomAccessDataset dataset);
    }
}
