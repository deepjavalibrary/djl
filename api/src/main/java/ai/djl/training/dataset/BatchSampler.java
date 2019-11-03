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

/** Wraps another subSampler to yield a mini-batch of indices. */
public class BatchSampler implements Sampler {

    private Sampler.SubSampler subSampler;
    private long batchSize;
    private boolean dropLast;

    public BatchSampler(Sampler.SubSampler subSampler, long batchSize) {
        this(subSampler, batchSize, false);
    }

    public BatchSampler(Sampler.SubSampler subSampler, long batchSize, boolean dropLast) {
        this.subSampler = subSampler;
        this.batchSize = batchSize;
        this.dropLast = dropLast;
    }

    /** {@inheritDoc} */
    @Override
    public Iterator<List<Long>> sample(RandomAccessDataset dataset) {
        return new Iterate(dataset);
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
