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

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Transform;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.util.Progress;
import ai.djl.util.RandomUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * RandomAccessDataset represent the dataset that support random access reads. i.e. it could access
 * a specific data item given the index.
 *
 * <p>Almost all datasets in DJL extend, either directly or indirectly, {@link RandomAccessDataset}.
 *
 * @see <a href="https://docs.djl.ai/master/docs/development/how_to_use_dataset.html">The guide to
 *     implementing a custom dataset</a>
 */
public abstract class RandomAccessDataset implements Dataset {

    protected Sampler sampler;
    protected Batchifier dataBatchifier;
    protected Batchifier labelBatchifier;
    protected Pipeline pipeline;
    protected Pipeline targetPipeline;
    protected int prefetchNumber;
    protected long limit;
    protected Device device;

    RandomAccessDataset() {}

    /**
     * Creates a new instance of {@link RandomAccessDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public RandomAccessDataset(BaseBuilder<?> builder) {
        this.sampler = builder.getSampler();
        this.dataBatchifier = builder.dataBatchifier;
        this.labelBatchifier = builder.labelBatchifier;
        this.pipeline = builder.pipeline;
        this.targetPipeline = builder.targetPipeline;
        this.prefetchNumber = builder.prefetchNumber;
        this.limit = builder.limit;
        this.device = builder.device;
    }

    /**
     * Gets the {@link Record} for the given index from the dataset.
     *
     * @param manager the manager used to create the arrays
     * @param index the index of the requested data item
     * @return a {@link Record} that contains the data and label of the requested data item
     * @throws IOException if an I/O error occurs
     */
    public abstract Record get(NDManager manager, long index) throws IOException;

    /** {@inheritDoc} */
    @Override
    public Iterable<Batch> getData(NDManager manager) throws IOException, TranslateException {
        return getData(manager, sampler, null);
    }

    /** {@inheritDoc} */
    @Override
    public Iterable<Batch> getData(NDManager manager, ExecutorService executorService)
            throws IOException, TranslateException {
        return getData(manager, sampler, executorService);
    }

    /**
     * Fetches an iterator that can iterate through the {@link Dataset} with a custom sampler.
     *
     * @param manager the manager to create the arrays
     * @param sampler the sampler to use to iterate through the dataset
     * @return an {@link Iterable} of {@link Batch} that contains batches of data from the dataset
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public Iterable<Batch> getData(NDManager manager, Sampler sampler)
            throws IOException, TranslateException {
        return getData(manager, sampler, null);
    }

    /**
     * Fetches an iterator that can iterate through the {@link Dataset} with a custom sampler
     * multi-threaded.
     *
     * @param manager the manager to create the arrays
     * @param sampler the sampler to use to iterate through the dataset
     * @param executorService the executorService to multi-thread with
     * @return an {@link Iterable} of {@link Batch} that contains batches of data from the dataset
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public Iterable<Batch> getData(
            NDManager manager, Sampler sampler, ExecutorService executorService)
            throws IOException, TranslateException {
        prepare();
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

    /**
     * Returns the size of this {@code Dataset}.
     *
     * @return the size of this {@code Dataset}
     */
    public long size() {
        return Math.min(limit, availableSize());
    }

    /**
     * Returns the number of records available to be read in this {@code Dataset}.
     *
     * @return the number of records available to be read in this {@code Dataset}
     */
    protected abstract long availableSize();

    /**
     * Splits the dataset set into multiple portions.
     *
     * @param ratio the ratio of each sub dataset
     * @return an array of the sub dataset
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public RandomAccessDataset[] randomSplit(int... ratio) throws IOException, TranslateException {
        prepare();
        if (ratio.length < 2) {
            throw new IllegalArgumentException("Requires at least two split portion.");
        }
        int size = Math.toIntExact(size());
        int[] indices = IntStream.range(0, size).toArray();
        for (int i = 0; i < size; ++i) {
            swap(indices, i, RandomUtils.nextInt(size));
        }
        RandomAccessDataset[] ret = new RandomAccessDataset[ratio.length];

        double sum = Arrays.stream(ratio).sum();
        int from = 0;
        for (int i = 0; i < ratio.length - 1; ++i) {
            int to = from + (int) (ratio[i] / sum * size);
            ret[i] = newSubDataset(indices, from, to);
            from = to;
        }
        ret[ratio.length - 1] = newSubDataset(indices, from, size);
        return ret;
    }

    /**
     * Returns a view of the portion of this data between the specified {@code fromIndex},
     * inclusive, and {@code toIndex}, exclusive.
     *
     * @param fromIndex low endpoint (inclusive) of the subDataset
     * @param toIndex high endpoint (exclusive) of the subData
     * @return a view of the specified range within this dataset
     */
    public RandomAccessDataset subDataset(int fromIndex, int toIndex) {
        int size = Math.toIntExact(size());
        int[] indices = IntStream.range(0, size).toArray();
        return newSubDataset(indices, fromIndex, toIndex);
    }

    /**
     * Returns a view of the portion of this data for the specified {@code subIndices}.
     *
     * @param subIndices sub-set of indices of this dataset
     * @return a view of the specified indices within this dataset
     */
    public RandomAccessDataset subDataset(List<Long> subIndices) {
        if (BulkDataIterable.isRange(subIndices)) {
            int size = Math.toIntExact(size());
            int[] indices = IntStream.range(0, size).toArray();
            long fromIndex = subIndices.get(0);
            long toIndex = subIndices.get(0) + subIndices.size();
            return newSubDataset(indices, Math.toIntExact(fromIndex), Math.toIntExact(toIndex));
        }
        return newSubDataset(subIndices);
    }

    /**
     * Returns a view of the portion of this data for the specified record keys. Assuming that the
     * records of this database are represented by the keys in <code>recordKeys</code>, then <code>
     * subRecordKeys</code> defines the view on the corresponding records of the database.
     *
     * @param recordKeys unique keys for all records of this dataset.
     * @param subRecordKeys keys to define the view on the dataset. All keys in <code>subRecordKeys
     *     </code> must be contained in <code>recordKeys</code> but may occur more than once.
     * @param <K> the record key type.
     * @return a view of the specified records within this dataset
     */
    public <K> RandomAccessDataset subDataset(List<K> recordKeys, List<K> subRecordKeys) {
        if (this.size() != recordKeys.size()) {
            throw new IllegalArgumentException(
                    "Requires as many record keys as there are records in the dataset.");
        }
        Map<K, Long> indicesOfRecordKeys = new ConcurrentHashMap<>(recordKeys.size());
        for (int index = 0; index < recordKeys.size(); index++) {
            Long prevIndex = indicesOfRecordKeys.put(recordKeys.get(index), (long) index);
            if (prevIndex != null) {
                throw new IllegalArgumentException(
                        "At least two keys at position "
                                + prevIndex
                                + " and "
                                + index
                                + " are equal!");
            }
        }
        return subDataset(indicesOfRecordKeys, subRecordKeys);
    }

    /**
     * Returns a view of the portion of this data for the specified record keys. Assuming that the
     * records of this database are represented by the keys in <code>indicesOfRecordKeys</code>,
     * then <code>
     * subRecordKeys</code> defines the view on the corresponding records of the database.
     *
     * @param indicesOfRecordKeys Map for keys of the records in this dataset to their index
     *     position within this dataset. While this map typically maps all records, technically it
     *     just needs to map the ones occurring in <code>subRecordKeys</code>.
     * @param subRecordKeys Keys to define the view on the dataset. All keys in <code>subRecordKeys
     *     </code> must be contained in <code>indicesOfRecordKeys</code> but may occur more than
     *     once.
     * @param <K> the record key type.
     * @return a view of the records identified by the specified keys of this dataset
     */
    public <K> RandomAccessDataset subDataset(
            Map<K, Long> indicesOfRecordKeys, List<K> subRecordKeys) {
        List<Long> subIndices = new ArrayList<>(subRecordKeys.size());
        for (K recordKey : subRecordKeys) {
            Long index = indicesOfRecordKeys.get(recordKey);
            if (index == null) {
                throw new IllegalArgumentException(
                        "The key of subRecordKeys at position "
                                + subRecordKeys.indexOf(recordKey)
                                + " is not contained in recordKeys!");
            }
            subIndices.add(index);
        }
        return subDataset(subIndices);
    }

    protected RandomAccessDataset newSubDataset(int[] indices, int from, int to) {
        return new SubDataset(this, indices, from, to);
    }

    protected RandomAccessDataset newSubDataset(List<Long> subIndices) {
        return new SubDatasetByIndices(this, subIndices);
    }

    /**
     * Returns the dataset contents as a Java array.
     *
     * <p>Each Number[] is a flattened dataset record and the Number[][] is the array of all
     * records.
     *
     * @param manager the manager to create the arrays
     * @return the dataset contents as a Java array
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public Pair<Number[][], Number[][]> toArray(NDManager manager)
            throws IOException, TranslateException {
        Sampler sampl = new BatchSampler(new SequenceSampler(), 1, false);
        int size = Math.toIntExact(size());
        Number[][] data = new Number[size][];
        Number[][] labels = new Number[size][];
        int index = 0;
        for (Batch batch : getData(manager, sampl)) {
            data[index] = flattenRecord(batch.getData());
            labels[index] = flattenRecord(batch.getLabels());
            batch.close();
            index++;
        }
        return new Pair<>(data, labels);
    }

    private Number[] flattenRecord(NDList data) {
        NDList flattened =
                new NDList(data.stream().map(NDArray::flatten).collect(Collectors.toList()));
        if (flattened.isEmpty()) {
            return null; // NOPMD
        }
        if (flattened.size() == 1) {
            return flattened.get(0).toArray();
        }
        return NDArrays.concat(flattened).toArray();
    }

    private static void swap(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    /** The Builder to construct a {@link RandomAccessDataset}. */
    public abstract static class BaseBuilder<T extends BaseBuilder<T>> {

        protected Sampler sampler;
        protected Batchifier dataBatchifier = Batchifier.STACK;
        protected Batchifier labelBatchifier = Batchifier.STACK;
        protected Pipeline pipeline;
        protected Pipeline targetPipeline;
        protected int prefetchNumber = 2;
        protected long limit = Long.MAX_VALUE;
        protected Device device;

        /**
         * Gets the {@link Sampler} for the dataset.
         *
         * @return the {@code Sampler}
         */
        public Sampler getSampler() {
            Objects.requireNonNull(sampler, "The sampler must be set");
            return sampler;
        }

        /**
         * Sets the {@link Sampler} with the given batch size.
         *
         * @param batchSize the batch size
         * @param random whether the sampling has to be random
         * @return this {@code BaseBuilder}
         */
        public T setSampling(int batchSize, boolean random) {
            return setSampling(batchSize, random, false);
        }

        /**
         * Sets the {@link Sampler} with the given batch size.
         *
         * @param batchSize the batch size
         * @param random whether the sampling has to be random
         * @param dropLast whether to drop the last incomplete batch
         * @return this {@code BaseBuilder}
         */
        public T setSampling(int batchSize, boolean random, boolean dropLast) {
            if (random) {
                sampler = new BatchSampler(new RandomSampler(), batchSize, dropLast);
            } else {
                sampler = new BatchSampler(new SequenceSampler(), batchSize, dropLast);
            }
            return self();
        }

        /**
         * Sets the {@link Sampler} for the dataset.
         *
         * @param sampler the {@link Sampler} to be set
         * @return this {@code BaseBuilder}
         */
        public T setSampling(Sampler sampler) {
            this.sampler = sampler;
            return self();
        }

        /**
         * Sets the {@link Batchifier} for the data.
         *
         * @param dataBatchifier the {@link Batchifier} to be set
         * @return this {@code BaseBuilder}
         */
        public T optDataBatchifier(Batchifier dataBatchifier) {
            this.dataBatchifier = dataBatchifier;
            return self();
        }

        /**
         * Sets the {@link Batchifier} for the labels.
         *
         * @param labelBatchifier the {@link Batchifier} to be set
         * @return this {@code BaseBuilder}
         */
        public T optLabelBatchifier(Batchifier labelBatchifier) {
            this.labelBatchifier = labelBatchifier;
            return self();
        }

        /**
         * Sets the {@link Pipeline} of {@link ai.djl.translate.Transform} to be applied on the
         * data.
         *
         * @param pipeline the {@link Pipeline} of {@link ai.djl.translate.Transform} to be applied
         *     on the data
         * @return this {@code BaseBuilder}
         */
        public T optPipeline(Pipeline pipeline) {
            this.pipeline = pipeline;
            return self();
        }

        /**
         * Adds the {@link Transform} to the {@link Pipeline} to be applied on the data.
         *
         * @param transform the {@link Transform} to be added
         * @return this builder
         */
        public T addTransform(Transform transform) {
            if (pipeline == null) {
                pipeline = new Pipeline();
            }
            pipeline.add(transform);
            return self();
        }

        /**
         * Sets the {@link Pipeline} of {@link ai.djl.translate.Transform} to be applied on the
         * labels.
         *
         * @param targetPipeline the {@link Pipeline} of {@link ai.djl.translate.Transform} to be
         *     applied on the labels
         * @return this {@code BaseBuilder}
         */
        public T optTargetPipeline(Pipeline targetPipeline) {
            this.targetPipeline = targetPipeline;
            return self();
        }

        /**
         * Adds the {@link Transform} to the target {@link Pipeline} to be applied on the labels.
         *
         * @param transform the {@link Transform} to be added
         * @return this builder
         */
        public T addTargetTransform(Transform transform) {
            if (targetPipeline == null) {
                targetPipeline = new Pipeline();
            }
            targetPipeline.add(transform);
            return self();
        }

        /**
         * Sets the number of batches to prefetch at once.
         *
         * @param prefetchNumber the number of batches to prefetch at once
         * @return this {@code BaseBuilder}
         */
        public T optPrefetchNumber(int prefetchNumber) {
            this.prefetchNumber = prefetchNumber;
            return self();
        }

        /**
         * Sets the {@link Device}.
         *
         * @param device the device
         * @return this {@code BaseBuilder}
         */
        public T optDevice(Device device) {
            this.device = device;
            return self();
        }

        /**
         * Sets this dataset's limit.
         *
         * <p>The limit is usually used for testing purposes to test only with a subset of the
         * dataset.
         *
         * @param limit the limit of this dataset's records
         * @return this {@code BaseBuilder}
         */
        public T optLimit(long limit) {
            this.limit = limit;
            return self();
        }

        /**
         * Returns this {code Builder} object.
         *
         * @return this {@code BaseBuilder}
         */
        protected abstract T self();
    }

    private static final class SubDataset extends RandomAccessDataset {

        private RandomAccessDataset dataset;
        private int[] indices;
        private int from;
        private int to;

        public SubDataset(RandomAccessDataset dataset, int[] indices, int from, int to) {
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
        public Record get(NDManager manager, long index) throws IOException {
            if (index >= size()) {
                throw new IndexOutOfBoundsException("index(" + index + ") > size(" + size() + ").");
            }
            return dataset.get(manager, indices[Math.toIntExact(index) + from]);
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

    private static final class SubDatasetByIndices extends RandomAccessDataset {

        private RandomAccessDataset dataset;
        private List<Long> subIndices;

        public SubDatasetByIndices(RandomAccessDataset dataset, List<Long> subIndices) {
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
        public Record get(NDManager manager, long index) throws IOException {
            return dataset.get(manager, subIndices.get(Math.toIntExact(index)));
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
