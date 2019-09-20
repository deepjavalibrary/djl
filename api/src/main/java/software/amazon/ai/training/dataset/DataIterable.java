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

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.Device;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.translate.TrainTranslator;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Pair;

// TODO abstract a interface that could be inherited by this and Stream DataIterable
// where the random reads is expensive
public class DataIterable<I, L> implements Iterable<Batch> {
    private RandomAccessDataset<I, L> dataset;
    private Trainer<I, L, ?> trainer;
    private Sampler sampler;
    private ExecutorService executor;
    private Device device;

    public DataIterable(
            RandomAccessDataset<I, L> dataset,
            Trainer<I, L, ?> trainer,
            Sampler sampler,
            ExecutorService executor,
            Device device) {
        this.dataset = dataset;
        this.trainer = trainer;
        this.sampler = sampler;
        this.executor = executor;
        this.device = device;
    }

    @Override
    public Iterator<Batch> iterator() {
        return new DataIterator<>(dataset, trainer, sampler, executor, device);
    }

    private static class DataIterator<I, L> implements Iterator<Batch> {
        private RandomAccessDataset<I, L> dataset;
        private Trainer<I, L, ?> trainer;
        private Iterator<List<Long>> sample;
        private ExecutorService executor;
        private Device device;
        // for multithreading
        private Queue<Future<Batch>> queue;
        private static final Logger logger = LoggerFactory.getLogger(DataIterable.class);

        public DataIterator(
                RandomAccessDataset<I, L> dataset,
                Trainer<I, L, ?> trainer,
                Sampler sampler,
                ExecutorService executor,
                Device device) {
            this.dataset = dataset;
            this.trainer = trainer;
            this.sample = sampler.sample(trainer, dataset);
            this.executor = executor;
            this.device = device;

            if (executor != null) {
                queue = new LinkedList<>();
                // prefetch
                for (int i = 0; i < ((ThreadPoolExecutor) executor).getCorePoolSize() * 2; i++) {
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
                List<Long> indices = sample.next();
                return fetch(indices);
            } else {
                preFetch();
                Future<Batch> future = queue.poll();
                try {
                    return future.get();
                } catch (InterruptedException | ExecutionException e) {
                    logger.error(e.getMessage());
                }
                throw new IllegalStateException("Data loading failed");
            }
        }

        private Batch fetch(List<Long> indices) {
            NDList[] data = new NDList[indices.size()];
            NDList[] labels = new NDList[indices.size()];
            TranslatorContext ctx = trainer.getPreprocessContext();
            TrainTranslator<I, L, ?> translator = trainer.getTranslator();
            for (int i = 0; i < indices.size(); i++) {
                Pair<I, L> dataItem = dataset.get(indices.get(i));
                Record record;
                try {
                    record = translator.processInput(ctx, dataItem);
                } catch (Exception e) {
                    throw new IllegalStateException("Failed to get next data item", e);
                }
                data[i] = record.getData();
                labels[i] = record.getLabels();
            }
            Batchifier batchifier = translator.getBatchifier();
            NDList batchData = batchifier.batchify(data);
            NDList batchLabels = batchifier.batchify(labels);
            // pin to a specific device
            if (device != null) {
                batchData = batchData.asInContext(device, false);
                batchLabels = batchLabels.asInContext(device, false);
            }
            Batch batch = new Batch(trainer.getManager(), batchData, batchLabels);
            ctx.close();
            return batch;
        }

        private void preFetch() {
            List<Long> indices;
            if (sample.hasNext()) {
                indices = sample.next();
            } else {
                return;
            }
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
            public Batch call() {
                return fetch(indices);
            }
        }
    }
}
