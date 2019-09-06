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

import java.io.IOException;
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
import software.amazon.ai.Device;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.translate.Pipeline;

// TODO abstract a interface that could be inherited by this and Stream DataIterable
// where the random reads is expensive
public class DataIterable implements Iterable<Batch> {
    private RandomAccessDataset dataset;
    private Sampler sampler;
    private Batchifier batchifier;
    private Pipeline pipeline;
    private Pipeline targetPipeline;
    private ExecutorService executor;
    private int preFetchNumber;
    private Device device;

    public DataIterable(
            RandomAccessDataset dataset,
            Sampler sampler,
            Batchifier batchifier,
            Pipeline pipeline,
            Pipeline targetPipeline,
            ExecutorService executor,
            int preFetchNumber,
            Device device) {
        this.dataset = dataset;
        this.sampler = sampler;
        this.batchifier = batchifier;
        this.pipeline = pipeline;
        this.targetPipeline = targetPipeline;
        this.executor = executor;
        this.preFetchNumber = preFetchNumber;
        this.device = device;
    }

    @Override
    public Iterator<Batch> iterator() {
        return new DataIterator(
                dataset,
                sampler,
                batchifier,
                pipeline,
                targetPipeline,
                executor,
                preFetchNumber,
                device);
    }

    private static class DataIterator implements Iterator<Batch> {
        private RandomAccessDataset dataset;
        private Iterator<List<Long>> sample;
        private Batchifier batchifier;
        private Pipeline pipeline;
        private Pipeline targetPipeline;
        private ExecutorService executor;
        private Device device;
        // for multithreading
        private Queue<Future<Batch>> queue;
        private static final Logger logger = LoggerFactory.getLogger(DataIterable.class);

        public DataIterator(
                RandomAccessDataset dataset,
                Sampler sampler,
                Batchifier batchifier,
                Pipeline pipeline,
                Pipeline targetPipeline,
                ExecutorService executor,
                int prefetchNumber,
                Device device) {
            this.dataset = dataset;
            this.sample = sampler.sample(dataset);
            this.batchifier = batchifier;
            this.pipeline = pipeline;
            this.targetPipeline = targetPipeline;
            this.executor = executor;
            this.device = device;

            if (executor != null) {
                queue = new LinkedList<>();
                // prefetch
                for (int i = 0; i < prefetchNumber; i++) {
                    preFetch();
                }
            }
        }

        @Override
        public boolean hasNext() {
            if (executor != null) {
                return !queue.isEmpty();
            }
            return sample.hasNext();
        }

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
            NDList[] data = new NDList[indices.size()];
            NDList[] labels = new NDList[indices.size()];
            for (int i = 0; i < indices.size(); i++) {
                Record record = dataset.get(indices.get(i));
                data[i] = record.getData();
                labels[i] = record.getLabels();
            }
            NDList batchData = batchifier.batchify(data);
            NDList batchLabels = batchifier.batchify(labels);
            // apply transform
            if (pipeline != null) {
                batchData = pipeline.transform(batchData, false);
            }
            // apply label transform
            if (targetPipeline != null) {
                batchLabels = targetPipeline.transform(batchLabels, false);
            }
            // pin to a specific device
            if (device != null) {
                batchData = batchData.asInDevice(device, false);
                batchLabels = batchLabels.asInDevice(device, false);
            }
            return new Batch(batchData, batchLabels);
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

            @Override
            public Batch call() throws IOException {
                return fetch(indices);
            }
        }
    }
}
