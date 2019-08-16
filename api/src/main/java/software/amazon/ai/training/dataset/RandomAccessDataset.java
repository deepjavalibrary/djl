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
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.util.Pair;

/**
 * RandomAccessDataset represent the dataset that support random access reads. i.e. it could access
 * certain data item given the index
 */
public abstract class RandomAccessDataset implements Dataset, RandomAccess {
    private long size;
    private DataLoadingConfiguration config;

    public RandomAccessDataset(DataLoadingConfiguration config) {
        this.config = config;
    }

    public abstract Pair<NDList, NDList> get(long index);

    @Override
    public Iterable<Record> getRecords() {
        return new DataIterable(this, config);
    }

    public long size() {
        return size;
    }

    protected void setSize(long size) {
        this.size = size;
    }

    protected DataLoadingConfiguration getDataLoadingConfiguration() {
        return config;
    }
}
