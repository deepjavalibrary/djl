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
import java.util.NoSuchElementException;

/**
 * {@code SequenceSampler} is an implementation of the {@link
 * ai.djl.training.dataset.Sampler.SubSampler} interface.
 *
 * <p>{@code SequenceSampler} samples the data from [0, dataset.size) sequentially.
 */
public class SequenceSampler implements Sampler.SubSampler {

    /** {@inheritDoc} */
    @Override
    public Iterator<Long> sample(RandomAccessDataset dataset) {
        return new Iterate(dataset);
    }

    static class Iterate implements Iterator<Long> {

        private long size;
        private long current;

        Iterate(RandomAccessDataset dataset) {
            size = dataset.size();
            current = 0;
        }

        /** {@inheritDoc} */
        @Override
        public boolean hasNext() {
            return current < size;
        }

        /** {@inheritDoc} */
        @Override
        public Long next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return current++;
        }
    }
}
