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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Progress;
import java.io.IOException;
import java.util.stream.Stream;

/**
 * {@code ArrayDataset} is an implementation of {@link RandomAccessDataset} that consist entirely of
 * large {@link NDArray}s. There can be multiple data and label {@link NDArray}s within the dataset.
 * Each sample will be retrieved by indexing each {@link NDArray} along the first dimension.
 *
 * <p>The following is an example of how to use ArrayDataset:
 *
 * <pre>
 *     ArrayDataset dataset = new ArrayDataset.Builder()
 *                              .setData(data)
 *                              .optLabels(label)
 *                              .setSampling(20, false)
 *                              .build();
 * </pre>
 *
 * @see Dataset
 */
public class ArrayDataset extends RandomAccessDataset {

    protected NDArray[] data;
    protected NDArray[] labels;

    /**
     * Creates a new instance of {@code ArrayDataset} with the arguments in {@link Builder}.
     *
     * @param builder a builder with the required arguments
     */
    public ArrayDataset(BaseBuilder<?> builder) {
        super(builder);
        if (builder instanceof Builder) {
            Builder builder2 = (Builder) builder;
            data = builder2.data;
            labels = builder2.labels;

            // check data and labels have the same size
            long size = data[0].size(0);
            if (Stream.of(data).anyMatch(array -> array.size(0) != size)) {
                throw new IllegalArgumentException("All the NDArray must have the same length!");
            }
            if (labels != null && Stream.of(labels).anyMatch(array -> array.size(0) != size)) {
                throw new IllegalArgumentException("All the NDArray must have the same length!");
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    protected long availableSize() {
        return data[0].size(0);
    }

    /** {@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) {
        NDList datum = new NDList();
        NDList label = new NDList();
        for (NDArray array : data) {
            datum.add(array.get(index));
        }
        if (labels != null) {
            for (NDArray array : labels) {
                label.add(array.get(index));
            }
        }
        datum.attach(manager);
        label.attach(manager);
        return new Record(datum, label);
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {}

    /** The Builder to construct an {@link ArrayDataset}. */
    public static final class Builder extends BaseBuilder<Builder> {

        private NDArray[] data;
        private NDArray[] labels;

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the data as an {@link NDArray} for the {@code ArrayDataset}.
         *
         * @param data an array of {@link NDArray} that contains the data
         * @return this Builder
         */
        public Builder setData(NDArray... data) {
            this.data = data;
            return self();
        }

        /**
         * Sets the labels for the data in the {@code ArrayDataset}.
         *
         * @param labels an array of {@link NDArray} that contains the labels
         * @return this Builder
         */
        public Builder optLabels(NDArray... labels) {
            this.labels = labels;
            return self();
        }

        /**
         * Builds a new instance of {@code ArrayDataset} with the specified data and labels.
         *
         * @return a new instance of {@code ArrayDataset}
         */
        public ArrayDataset build() {
            if (data == null || data.length == 0) {
                throw new IllegalArgumentException("Please pass in at least one data");
            }
            return new ArrayDataset(this);
        }
    }
}
