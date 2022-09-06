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
import ai.djl.ndarray.index.NDIndex;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.stream.Stream;

/**
 * {@code ArrayDataset} is an implementation of {@link RandomAccessDataset} that consist entirely of
 * large {@link NDArray}s. It is recommended only for datasets small enough to fit in memory that
 * come in array formats. Otherwise, consider directly using the {@link RandomAccessDataset}
 * instead.
 *
 * <p>There can be multiple data and label {@link NDArray}s within the dataset. Each sample will be
 * retrieved by indexing each {@link NDArray} along the first dimension.
 *
 * <p>The following is an example of how to use ArrayDataset:
 *
 * <pre>
 *     ArrayDataset dataset = new ArrayDataset.Builder()
 *                              .setData(data1, data2)
 *                              .optLabels(labels1, labels2, labels3)
 *                              .setSampling(20, false)
 *                              .build();
 * </pre>
 *
 * <p>Suppose you get a {@link Batch} from {@code trainer.iterateDataset(dataset)} or {@code
 * dataset.getData(manager)}. In the data of this batch, it will be an NDList with one NDArray for
 * each data input. In this case, it would be 2 arrays. Similarly, the labels would have 3 arrays.
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

    ArrayDataset() {}

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
            datum.add(array.get(manager, index));
        }
        if (labels != null) {
            for (NDArray array : labels) {
                label.add(array.get(manager, index));
            }
        }
        return new Record(datum, label);
    }

    /**
     * Gets the {@link Batch} for the given indices from the dataset.
     *
     * @param manager the manager used to create the arrays
     * @param indices indices of the requested data items
     * @return a {@link Batch} that contains the data and label of the requested data items
     */
    public Batch getByIndices(NDManager manager, long... indices) {
        try (NDArray ndIndices = manager.create(indices)) {
            NDIndex index = new NDIndex("{}", ndIndices);
            NDList datum = new NDList();
            NDList label = new NDList();
            for (NDArray array : data) {
                datum.add(array.get(manager, index));
            }
            if (labels != null) {
                for (NDArray array : labels) {
                    label.add(array.get(manager, index));
                }
            }
            return new Batch(
                    manager,
                    datum,
                    label,
                    indices.length,
                    Batchifier.STACK,
                    Batchifier.STACK,
                    -1,
                    -1);
        }
    }

    /**
     * Gets the {@link Batch} for the given range from the dataset.
     *
     * @param manager the manager used to create the arrays
     * @param fromIndex low endpoint (inclusive) of the dataset
     * @param toIndex high endpoint (exclusive) of the dataset
     * @return a {@link Batch} that contains the data and label of the requested data items
     */
    public Batch getByRange(NDManager manager, long fromIndex, long toIndex) {
        NDIndex index = new NDIndex().addSliceDim(fromIndex, toIndex);
        NDList datum = new NDList();
        NDList label = new NDList();
        for (NDArray array : data) {
            datum.add(array.get(manager, index));
        }
        if (labels != null) {
            for (NDArray array : labels) {
                label.add(array.get(manager, index));
            }
        }
        int size = Math.toIntExact(toIndex - fromIndex);
        return new Batch(manager, datum, label, size, Batchifier.STACK, Batchifier.STACK, -1, -1);
    }

    /** {@inheritDoc} */
    @Override
    protected RandomAccessDataset newSubDataset(int[] indices, int from, int to) {
        return new SubDataset(this, indices, from, to);
    }

    @Override
    protected RandomAccessDataset newSubDataset(List<Long> subIndices) {
        return new SubDatasetByIndices(this, subIndices);
    }

    /** {@inheritDoc} */
    @Override
    public Iterable<Batch> getData(
            NDManager manager, Sampler sampler, ExecutorService executorService)
            throws IOException, TranslateException {
        prepare();
        if (dataBatchifier == Batchifier.STACK && labelBatchifier == Batchifier.STACK) {
            return new BulkDataIterable(
                    this,
                    manager,
                    sampler,
                    dataBatchifier,
                    labelBatchifier,
                    pipeline,
                    targetPipeline,
                    executorService,
                    prefetchNumber,
                    device);
        }
        return new DataIterable(
                this,
                manager,
                sampler,
                dataBatchifier,
                labelBatchifier,
                pipeline,
                targetPipeline,
                executorService,
                prefetchNumber,
                device);
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

    private static final class SubDataset extends ArrayDataset {

        private ArrayDataset dataset;
        private int[] indices;
        private int from;
        private int to;

        public SubDataset(ArrayDataset dataset, int[] indices, int from, int to) {
            this.dataset = dataset;
            this.indices = indices;
            this.from = from;
            this.to = to;
            this.sampler = dataset.sampler;
            this.dataBatchifier = dataset.dataBatchifier;
            this.labelBatchifier = dataset.labelBatchifier;
            this.pipeline = dataset.pipeline;
            this.targetPipeline = dataset.targetPipeline;
            this.prefetchNumber = dataset.prefetchNumber;
            this.device = dataset.device;

            limit = Long.MAX_VALUE;
        }

        /** {@inheritDoc} */
        @Override
        public Record get(NDManager manager, long index) {
            if (index >= size()) {
                throw new IndexOutOfBoundsException("index(" + index + ") > size(" + size() + ").");
            }
            return dataset.get(manager, indices[Math.toIntExact(index) + from]);
        }

        /** {@inheritDoc} */
        @Override
        public Batch getByIndices(NDManager manager, long... indices) {
            long[] resolvedIndices = new long[indices.length];
            int i = 0;
            for (long index : indices) {
                resolvedIndices[i++] = this.indices[Math.toIntExact(index) + from];
            }
            return dataset.getByIndices(manager, resolvedIndices);
        }

        /** {@inheritDoc} */
        @Override
        public Batch getByRange(NDManager manager, long fromIndex, long toIndex) {
            return dataset.getByRange(manager, fromIndex + from, toIndex + from);
        }

        /** {@inheritDoc} */
        @Override
        protected long availableSize() {
            return to - from;
        }

        /** {@inheritDoc} */
        @Override
        public void prepare(Progress progress) {}
    }

    private static final class SubDatasetByIndices extends ArrayDataset {

        private ArrayDataset dataset;
        private List<Long> subIndices;

        public SubDatasetByIndices(ArrayDataset dataset, List<Long> subIndices) {
            this.dataset = dataset;
            this.subIndices = subIndices;
            this.sampler = dataset.sampler;
            this.dataBatchifier = dataset.dataBatchifier;
            this.labelBatchifier = dataset.labelBatchifier;
            this.pipeline = dataset.pipeline;
            this.targetPipeline = dataset.targetPipeline;
            this.prefetchNumber = dataset.prefetchNumber;
            this.device = dataset.device;

            limit = Long.MAX_VALUE;
        }

        /** {@inheritDoc} */
        @Override
        public Record get(NDManager manager, long index) {
            return dataset.get(manager, subIndices.get(Math.toIntExact(index)));
        }

        /** {@inheritDoc} */
        @Override
        public Batch getByIndices(NDManager manager, long... indices) {
            long[] resolvedIndices = new long[indices.length];
            int i = 0;
            for (long index : indices) {
                resolvedIndices[i++] = subIndices.get(Math.toIntExact(index));
            }
            return dataset.getByIndices(manager, resolvedIndices);
        }

        /** {@inheritDoc} */
        @Override
        public Batch getByRange(NDManager manager, long fromIndex, long toIndex) {
            long[] resolvedIndices = new long[(int) (toIndex - fromIndex)];
            int i = 0;
            for (long index = fromIndex; index < toIndex; index++) {
                resolvedIndices[i++] = subIndices.get(Math.toIntExact(index));
            }
            return dataset.getByIndices(manager, resolvedIndices);
        }

        /** {@inheritDoc} */
        @Override
        protected long availableSize() {
            return subIndices.size();
        }

        /** {@inheritDoc} */
        @Override
        public void prepare(Progress progress) {}
    }
}
