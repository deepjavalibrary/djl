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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Pipeline;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * DataIterable is a data loader that combines {@link Dataset}, {@link Batchifier}, {@link
 * Pipeline}, and {@link Sampler} to provide an iterable over the given {@link RandomAccessDataset}.
 *
 * <p>We don't recommended using DataIterable directly. Instead use {@link RandomAccessDataset}
 * combined with {@link ai.djl.training.Trainer} to iterate over the {@link RandomAccessDataset}}
 */
public class DataIterable implements Iterable<Batch>, Iterator<Batch> {

    private static final Logger logger = LoggerFactory.getLogger(DataIterable.class);

    private RandomAccessDataset dataset;
    private NDManager manager;
    private Batchifier batchifier;
    private Pipeline pipeline;
    private Pipeline targetPipeline;
    private ExecutorService executor;
    private long maxIteration;
    private Device device;

    private Iterator<List<Long>> sample;
    // for multithreading
    private Queue<Future<Batch>> queue;
    private long count;

    /**
     * Creates a new instance of {@code DataIterable} with the given parameters.
     *
     * @param dataset the dataset to iterate on
     * @param manager the manager to create the arrays
     * @param sampler a sampler to sample data with
     * @param batchifier a batchifier
     * @param pipeline the pipeline of transforms to apply on the data
     * @param targetPipeline the pipeline of transforms to apply on the labels
     * @param executor an {@link ExecutorService}
     * @param preFetchNumber the number of samples to prefetch
     * @param maxIteration the maximum number of iterations
     * @param device the {@link Device}
     */
    public DataIterable(
            RandomAccessDataset dataset,
            NDManager manager,
            Sampler sampler,
            Batchifier batchifier,
            Pipeline pipeline,
            Pipeline targetPipeline,
            ExecutorService executor,
            int preFetchNumber,
            long maxIteration,
            Device device) {
        this.dataset = dataset;
        this.manager = manager.newSubManager();
        this.batchifier = batchifier;
        this.pipeline = pipeline;
        this.targetPipeline = targetPipeline;
        this.executor = executor;
        this.maxIteration = maxIteration;
        this.device = device;

        sample = sampler.sample(dataset);
        if (executor != null) {
            queue = new LinkedList<>();
            // prefetch
            for (int i = 0; i < preFetchNumber; i++) {
                preFetch();
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public Iterator<Batch> iterator() {
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasNext() {
        if (++count > maxIteration) {
            return false;
        }

        if (executor != null) {
            if (queue.isEmpty()) {
                manager.close();
                return false;
            }
            return true;
        }
        if (!sample.hasNext()) {
            manager.close();
            return false;
        }
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public Batch next() {
        if (executor == null) {
            // single thread data loading with blocking fetch
            List<Long> indices = sample.next();
            try {
                return fetch(indices);
            } catch (IOException e) {
                logger.error(e.getMessage());
                throw new IllegalStateException("Data loading failed", e);
            }
        } else {
            // multithreading data loading with async fetch
            preFetch();
            Future<Batch> future = queue.poll();
            try {
                return future.get();
            } catch (InterruptedException | ExecutionException e) {
                logger.error(e.getMessage());
                throw new IllegalStateException("Data loading failed", e);
            }
        }
    }

    private Batch fetch(List<Long> indices) throws IOException {
        NDManager subManager = manager.newSubManager();
        NDList[] data = new NDList[indices.size()];
        NDList[] labels = new NDList[indices.size()];
        for (int i = 0; i < indices.size(); i++) {
            Record record = dataset.get(subManager, indices.get(i));
            data[i] = record.getData();
            // apply transform
            if (pipeline != null) {
                data[i] = pipeline.transform(data[i]);
            }

            labels[i] = record.getLabels();
        }
        NDList batchData = batchifier.batchify(data);
        NDList batchLabels = batchifier.batchify(labels);

        Arrays.stream(data).forEach(NDList::close);
        Arrays.stream(labels).forEach(NDList::close);

        // apply label transform
        if (targetPipeline != null) {
            batchLabels = targetPipeline.transform(batchLabels);
        }
        // pin to a specific device
        if (device != null) {
            batchData = batchData.asInDevice(device, false);
            batchLabels = batchLabels.asInDevice(device, false);
        }
        return new Batch(subManager, batchData, batchLabels, batchifier);
    }

    private void preFetch() {
        List<Long> indices;
        if (!sample.hasNext()) {
            return;
        }
        indices = sample.next();
        Callable<Batch> task = new PreFetchCallable(indices);
        Future<Batch> result = executor.submit(task);
        queue.offer(result);
    }

    class PreFetchCallable implements Callable<Batch> {

        private List<Long> indices;

        public PreFetchCallable(List<Long> indices) {
            this.indices = indices;
        }

        /** {@inheritDoc} */
        @Override
        public Batch call() throws IOException {
            return fetch(indices);
        }
    }
}
