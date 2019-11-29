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

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * {@code BatchSampler} is a {@link Sampler} that returns a single epoch over the data.
 *
 * <p>{@code BatchSampler} wraps another {@link ai.djl.training.dataset.Sampler.SubSampler} to yield
 * a mini-batch of indices.
 */
public class BatchSampler implements Sampler {

    private Sampler.SubSampler subSampler;
    private int batchSize;
    private boolean dropLast;

    /**
     * Creates a new instance of {@code BatchSampler} that samples from the given {@link
     * ai.djl.training.dataset.Sampler.SubSampler}, and yields a mini-batch of indices.
     *
     * <p>The last batch will not be dropped. The size of the last batch maybe smaller than batch
     * size in case the size of the dataset is not a multiple of batch size.
     *
     * @param subSampler the {@link ai.djl.training.dataset.Sampler.SubSampler} to sample from
     * @param batchSize the required batch size
     */
    public BatchSampler(Sampler.SubSampler subSampler, int batchSize) {
        this(subSampler, batchSize, false);
    }

    /**
     * Creates a new instance of {@code BatchSampler} that samples from the given {@link
     * ai.djl.training.dataset.Sampler.SubSampler}, and yields a mini-batch of indices.
     *
     * @param subSampler the {@link ai.djl.training.dataset.Sampler.SubSampler} to sample from
     * @param batchSize the required batch size
     * @param dropLast whether the {@code BatchSampler} should drop the last few samples in case the
     *     size of the dataset is not a multiple of batch size
     */
    public BatchSampler(Sampler.SubSampler subSampler, int batchSize, boolean dropLast) {
        this.subSampler = subSampler;
        this.batchSize = batchSize;
        this.dropLast = dropLast;
    }

    /** {@inheritDoc} */
    @Override
    public Iterator<List<Long>> sample(RandomAccessDataset dataset) {
        return new Iterate(dataset);
    }

    /** {@inheritDoc} */
    @Override
    public int getBatchSize() {
        return batchSize;
    }

    class Iterate implements Iterator<List<Long>> {

        private long size;
        private long current;
        private Iterator<Long> itemSampler;

        Iterate(RandomAccessDataset dataset) {
            current = 0;
            if (dropLast) {
                this.size = dataset.size() / batchSize;
            } else {
                this.size = (dataset.size() + batchSize - 1) / batchSize;
            }
            itemSampler = subSampler.sample(dataset);
        }

        /** {@inheritDoc} */
        @Override
        public boolean hasNext() {
            return current < size;
        }

        /** {@inheritDoc} */
        @Override
        public List<Long> next() {
            List<Long> batchIndices = new ArrayList<>();
            while (itemSampler.hasNext()) {
                batchIndices.add(itemSampler.next());
                if (batchIndices.size() == batchSize) {
                    break;
                }
            }
            current++;
            return batchIndices;
        }
    }
}
