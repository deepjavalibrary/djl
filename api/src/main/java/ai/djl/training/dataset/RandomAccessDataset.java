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
import java.io.IOException;
import java.util.RandomAccess;
import java.util.concurrent.ExecutorService;

/**
 * RandomAccessDataset represent the dataset that support random access reads. i.e. it could access
 * a specific data item given the index.
 */
public abstract class RandomAccessDataset implements Dataset, RandomAccess {

    protected Sampler sampler;
    protected Batchifier batchifier;
    protected Pipeline pipeline;
    protected Pipeline targetPipeline;
    protected ExecutorService executor;
    protected int prefetchNumber;
    private long maxIteration;
    protected Device device;

    /**
     * Creates a new instance of {@link RandomAccessDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public RandomAccessDataset(BaseBuilder<?> builder) {
        this.sampler = builder.getSampler();
        this.batchifier = builder.batchifier;
        this.pipeline = builder.pipeline;
        this.targetPipeline = builder.targetPipeline;
        this.executor = builder.executor;
        this.prefetchNumber = builder.prefetchNumber;
        this.maxIteration = builder.maxIteration;
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
    public Iterable<Batch> getData(NDManager manager) {
        return new DataIterable(
                this,
                manager,
                sampler,
                batchifier,
                pipeline,
                targetPipeline,
                executor,
                prefetchNumber,
                maxIteration,
                device);
    }

    /**
     * Returns the size of this {@code Dataset}.
     *
     * @return the size of this {@code Dataset}
     */
    public abstract long size();

    /** The Builder to construct a {@link RandomAccessDataset}. */
    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        protected Sampler sampler;
        protected Batchifier batchifier = Batchifier.STACK;
        protected Pipeline pipeline;
        protected Pipeline targetPipeline;
        protected ExecutorService executor;
        protected int prefetchNumber;
        protected long maxIteration = Long.MAX_VALUE;
        protected Device device;

        /**
         * Gets the {@link Sampler} for the dataset.
         *
         * @return the {@code Sampler}
         */
        public Sampler getSampler() {
            if (sampler == null) {
                throw new IllegalArgumentException("The sampler must be set");
            }
            return sampler;
        }

        /**
         * Sets the {@link Sampler} with the given batch size.
         *
         * @param batchSize the batch size
         * @param random whether the sampling has to be random
         * @return this {@code BaseBuilder}
         */
        public T setSampling(long batchSize, boolean random) {
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
        public T setSampling(long batchSize, boolean random, boolean dropLast) {
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
         * Sets the {@link Batchifier} for the dataset.
         *
         * @param batchier the {@link Batchifier} to be set
         * @return this {@code BaseBuilder}
         */
        public T optBatchier(Batchifier batchier) {
            this.batchifier = batchier;
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
         * Sets the {@link ExecutorService} to spawn threads to fetch data.
         *
         * @param executor the {@link ExecutorService} to spawn threads
         * @param prefetchNumber the number of samples to prefetch at once
         * @return this {@code BaseBuilder}
         */
        public T optExcutor(ExecutorService executor, int prefetchNumber) {
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
         * Sets the maximum number of iterations.
         *
         * @param maxIteration the maximum number of iterations
         * @return this {@code BaseBuilder}
         */
        public T optMaxIteration(long maxIteration) {
            this.maxIteration = maxIteration;
            return self();
        }

        /**
         * Returns this {code Builder} object.
         *
         * @return this {@code BaseBuilder}
         */
        protected abstract T self();
    }
}
