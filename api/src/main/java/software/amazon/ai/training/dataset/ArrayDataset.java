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

import java.util.stream.Stream;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.util.Pair;

/**
 * Dataset wrapping {@link NDArray}s. It is able to combine multiple data and labels. Each sample
 * will be retrieved by indexing {@link NDArray}s along the first dimension.
 *
 * <p>The following is an example of how to use ArrayDataset:
 *
 * <pre>
 *     ArrayDataset = new ArrayDataset.Builder()
 *             .setData(data1, data2)
 *             .setLabels(label1, label2, label3)
 *             .setDataLoadingProperty(false, 20, false)
 *             .build();
 * </pre>
 *
 * @see Dataset
 */
public final class ArrayDataset implements RandomAccessDataset {

    private final NDArray[] data;
    private final NDArray[] labels;
    private final Long size;
    private final DataLoadingConfiguration config;

    private ArrayDataset(Builder builder) {
        this.data = builder.data;
        this.labels = builder.labels;
        this.size = builder.size;
        this.config = builder.config;
    }

    /** {@inheritDoc} */
    @Override
    public Pair<NDList, NDList> get(long index) {
        NDList datum = new NDList();
        NDList label = new NDList();
        for (NDArray array : data) {
            datum.add(array.get(index));
        }
        for (NDArray array : labels) {
            label.add(array.get(index));
        }
        return new Pair<>(datum, label);
    }

    /** {@inheritDoc} */
    @Override
    public long size() {
        return size;
    }

    /** {@inheritDoc} */
    @Override
    public Iterable<Record> getRecords() {
        return new DataIterable(this, config);
    }

    public static final class Builder {

        private NDArray[] data;
        private NDArray[] labels;
        private DataLoadingConfiguration config;
        private Long size;

        public Builder() {
            this.data = new NDArray[0];
            this.labels = new NDArray[0];
        }

        public Builder setData(NDArray... data) {
            if (size == null) {
                size = data[0].size(0);
            }
            if (Stream.of(data).anyMatch(array -> array.size(0) != size)) {
                throw new IllegalArgumentException("All the NDArray must have the same length!");
            }
            this.data = data;
            return this;
        }

        public Builder setLabels(NDArray... labels) {
            if (size == null) {
                size = labels[0].size(0);
            }
            if (Stream.of(labels).anyMatch(array -> array.size(0) != size)) {
                throw new IllegalArgumentException("All the NDArray must have the same length!");
            }
            this.labels = labels;
            return this;
        }
        // TODO overload this function for other common params combination
        public Builder setDataLoadingProperty(boolean shuffle, int batchSize, boolean dropLast) {
            this.config =
                    new DataLoadingConfiguration.Builder()
                            .setShuffle(shuffle)
                            .setBatchSize(batchSize)
                            .setDropLast(dropLast)
                            .build();
            return this;
        }

        public Builder setDataLoadingProperty(DataLoadingConfiguration config) {
            if (this.config != null) {
                throw new IllegalArgumentException(
                        "either setDataLoading or setDataLoadingConfig, not both");
            }
            this.config = config;
            return this;
        }

        public ArrayDataset build() {
            if (this.config == null) {
                this.config =
                        new DataLoadingConfiguration.Builder()
                                .setShuffle(false)
                                .setBatchSize(1)
                                .setDropLast(false)
                                .build();
            }
            return new ArrayDataset(this);
        }
    }
}
