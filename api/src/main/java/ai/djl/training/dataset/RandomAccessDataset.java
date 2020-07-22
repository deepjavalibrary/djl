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
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Transform;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;
import ai.djl.util.RandomUtils;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.stream.IntStream;

/**
 * RandomAccessDataset represent the dataset that support random access reads. i.e. it could access
 * a specific data item given the index.
 */
public abstract class RandomAccessDataset implements Dataset {

    protected Sampler sampler;
    protected Batchifier dataBatchifier;
    protected Batchifier labelBatchifier;
    protected Pipeline pipeline;
    protected Pipeline targetPipeline;
    protected ExecutorService executor;
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
        this.executor = builder.executor;
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
    protected abstract Record get(NDManager manager, long index) throws IOException;

    /** {@inheritDoc} */
    @Override
    public Iterable<Batch> getData(NDManager manager) throws IOException, TranslateException {
        prepare();
        return new DataIterable(
                this,
                manager,
                sampler,
                dataBatchifier,
                labelBatchifier,
                pipeline,
                targetPipeline,
                executor,
                prefetchNumber,
                device);
    }

    /**
     * Fetches an iterator that can iterate through the {@link Dataset} with a custom sampler.
     *
     * @param manager the dataset to iterate through
     * @param sampler the sampler to use to iterate through the dataset
     * @return an {@link Iterable} of {@link Batch} that contains batches of data from the dataset
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public Iterable<Batch> getData(NDManager manager, Sampler sampler)
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
                executor,
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
            ret[i] = new SubDataset(this, indices, from, to);
            from += to;
        }
        ret[ratio.length - 1] = new SubDataset(this, indices, from, size);
        return ret;
    }

    private static void swap(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    /** The Builder to construct a {@link RandomAccessDataset}. */
    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        protected Sampler sampler;
        protected Batchifier dataBatchifier = Batchifier.STACK;
        protected Batchifier labelBatchifier = Batchifier.STACK;
        protected Pipeline pipeline;
        protected Pipeline targetPipeline;
        protected ExecutorService executor;
        protected int prefetchNumber;
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
         * Sets the {@link ExecutorService} to spawn threads to fetch data.
         *
         * @param executor the {@link ExecutorService} to spawn threads
         * @param prefetchNumber the number of samples to prefetch at once
         * @return this {@code BaseBuilder}
         */
        public T optExecutor(ExecutorService executor, int prefetchNumber) {
            this.executor = executor;
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
        public Iterable<Batch> getData(NDManager manager) throws IOException, TranslateException {
            return dataset.getData(manager);
        }

        /** {@inheritDoc} */
        @Override
        public void prepare(Progress progress) {}
    }
}
