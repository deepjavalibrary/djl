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

import java.util.RandomAccess;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.util.Pair;

/**
 * RandomAccessDataset represent the dataset that support random access reads. i.e. it could access
 * certain data item given the index
 */
public abstract class RandomAccessDataset<I, L> implements Dataset<I, L>, RandomAccess {

    protected long size;
    protected Sampler sampler;
    protected DataLoadingConfiguration config;

    public RandomAccessDataset(Sampler sampler, DataLoadingConfiguration config) {
        this.sampler = sampler;
        this.config = config;
    }

    public abstract Pair<I, L> get(long index);

    @Override
    public Iterable<Record> getRecords(Trainer<I, L, ?> trainer) {
        return new DataIterable<>(this, trainer, sampler, config);
    }

    public long size() {
        return size;
    }
}
