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

import java.util.ArrayList;
import java.util.List;

/** Wraps another sampler to yield a mini-batch of indices. */
public class BatchSampler implements Sampler<List<Long>> {

    private Sampler<Long> sampler;
    private long batchSize;
    private long current;
    private long size;

    public BatchSampler(Sampler<Long> sampler, long batchSize, boolean dropLast) {
        this.sampler = sampler;
        this.batchSize = batchSize;
        this.current = 0;
        if (dropLast) {
            this.size = sampler.size() / batchSize;
        } else {
            this.size = (sampler.size() + batchSize - 1) / batchSize;
        }
    }

    @Override
    public boolean hasNext() {
        return current < size;
    }

    @Override
    public List<Long> next() {
        List<Long> batchIndices = new ArrayList<>();
        while (sampler.hasNext()) {
            batchIndices.add(sampler.next());
            if (batchIndices.size() == batchSize) {
                break;
            }
        }
        current++;
        return batchIndices;
    }

    @Override
    public long size() {
        return this.size;
    }
}
