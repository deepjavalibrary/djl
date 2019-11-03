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
 * certain data item given the index
 */
public abstract class RandomAccessDataset implements Dataset, RandomAccess {

    protected long size;
    protected Sampler sampler;
    protected Batchifier batchifier;
    protected Pipeline pipeline;
    protected Pipeline targetPipeline;
    protected ExecutorService executor;
    protected int prefetchNumber;
    private long maxIteration;
    protected Device device;

    public RandomAccessDataset(BaseBuilder<?> builder) {
        this.sampler = builder.sampler;
        this.batchifier = builder.batchifier;
        this.pipeline = builder.pipeline;
        this.targetPipeline = builder.targetPipeline;
        this.executor = builder.executor;
        this.prefetchNumber = builder.prefetchNumber;
        this.maxIteration = builder.maxIteration;
        this.device = builder.device;
    }

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

    public long size() {
        return size;
    }

    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        Sampler sampler;
        Batchifier batchifier = Batchifier.STACK;
        Pipeline pipeline;
        Pipeline targetPipeline;
        ExecutorService executor;
        int prefetchNumber;
        long maxIteration = Long.MAX_VALUE;
        Device device;

        public Sampler getSampler() {
            if (sampler == null) {
                throw new IllegalArgumentException("The sampler must be set");
            }
            return sampler;
        }

        public T setRandomSampling(long batchSize) {
            sampler = new BatchSampler(new RandomSampler(), batchSize, true);
            return self();
        }

        public T setSequenceSampling(long batchSize) {
            return setSequenceSampling(batchSize, false);
        }

        public T setSequenceSampling(long batchSize, boolean dropLast) {
            sampler = new BatchSampler(new SequenceSampler(), batchSize, dropLast);
            return self();
        }

        public T setSampler(Sampler sampler) {
            this.sampler = sampler;
            return self();
        }

        public T optBatchier(Batchifier batchier) {
            this.batchifier = batchier;
            return self();
        }

        public T optPipeline(Pipeline pipeline) {
            this.pipeline = pipeline;
            return self();
        }

        public T optTargetPipeline(Pipeline targetPipeline) {
            this.targetPipeline = targetPipeline;
            return self();
        }

        public T optExcutor(ExecutorService executor, int prefetchNumber) {
            this.executor = executor;
            this.prefetchNumber = prefetchNumber;
            return self();
        }

        public T optDevice(Device device) {
            this.device = device;
            return self();
        }

        public T optMaxIteration(long maxIteration) {
            this.maxIteration = maxIteration;
            return self();
        }

        protected abstract T self();
    }
}
