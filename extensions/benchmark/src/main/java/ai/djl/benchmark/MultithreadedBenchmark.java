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
package ai.djl.benchmark;

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.listener.MemoryTrainingListener;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class runs single threaded benchmark. */
public class MultithreadedBenchmark extends AbstractBenchmark {

    private static final Logger logger = LoggerFactory.getLogger(MultithreadedBenchmark.class);

    /** {@inheritDoc} */
    @Override
    public float[] predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelException, TranslateException {

        MemoryTrainingListener.collectMemoryInfo(metrics); // Measure memory before loading model

        Engine engine = Engine.getEngine(arguments.getEngine());
        Device[] devices = engine.getDevices(arguments.getMaxGpus());
        int numOfThreads = arguments.getThreads();
        int delay = arguments.getDelay();
        AtomicInteger counter = new AtomicInteger(iteration);
        logger.info("Multithreading inference with {} threads.", numOfThreads);

        List<ZooModel<Void, float[]>> models = new ArrayList<>(devices.length);
        List<PredictorCallable> callables = new ArrayList<>(numOfThreads);
        for (Device device : devices) {
            ZooModel<Void, float[]> model = loadModel(arguments, metrics, device);
            models.add(model);

            for (int i = 0; i < numOfThreads / devices.length; ++i) {
                callables.add(new PredictorCallable(model, metrics, counter, i, i == 0));
            }
        }

        float[] result = null;
        ExecutorService executorService = Executors.newFixedThreadPool(numOfThreads);

        MemoryTrainingListener.collectMemoryInfo(metrics); // Measure memory before worker kickoff

        int successThreads = 0;
        try {
            for (PredictorCallable callable : callables) {
                callable.warmup();
            }

            metrics.addMetric("start", System.currentTimeMillis(), "mills");
            try {
                List<Future<float[]>> futures;
                if (delay > 0) {
                    futures = new ArrayList<>();
                    for (PredictorCallable callable : callables) {
                        futures.add(executorService.submit(callable));
                        Thread.sleep(delay);
                    }
                } else {
                    futures = executorService.invokeAll(callables);
                }

                for (Future<float[]> future : futures) {
                    result = future.get();
                    if (result != null) {
                        ++successThreads;
                    }
                }
            } catch (InterruptedException | ExecutionException e) {
                logger.error("", e);
            }
            metrics.addMetric("end", System.currentTimeMillis(), "mills");
            for (PredictorCallable callable : callables) {
                callable.close();
            }
        } finally {
            executorService.shutdown();
        }

        models.forEach(ZooModel::close);
        if (successThreads != numOfThreads) {
            logger.error("Only {}/{} threads finished.", successThreads, numOfThreads);
            return null;
        }

        return result;
    }

    private static class PredictorCallable implements Callable<float[]> {

        private Predictor<Void, float[]> predictor;

        private Metrics metrics;
        private String workerId;
        private boolean collectMemory;
        private AtomicInteger counter;
        private int total;
        private int steps;

        public PredictorCallable(
                ZooModel<Void, float[]> model,
                Metrics metrics,
                AtomicInteger counter,
                int workerId,
                boolean collectMemory) {
            this.predictor = model.newPredictor();
            this.metrics = metrics;
            this.counter = counter;
            this.workerId = String.format("%02d", workerId);
            this.collectMemory = collectMemory;
            predictor.setMetrics(metrics);
            total = counter.get();
            if (total < 10) {
                steps = 1;
            } else {
                steps = (int) Math.pow(10, (int) Math.log10(total));
            }
        }

        /** {@inheritDoc} */
        @Override
        public float[] call() throws Exception {
            float[] result = null;
            int count = 0;
            int remaining;
            while ((remaining = counter.decrementAndGet()) > 0 || result == null) {
                try {
                    result = predictor.predict(null);
                } catch (Exception e) {
                    // stop immediately when we find any exception
                    counter.set(0);
                    throw e;
                }
                if (collectMemory) {
                    MemoryTrainingListener.collectMemoryInfo(metrics);
                }
                int processed = total - remaining + 1;
                logger.trace("Worker-{}: {} iteration finished.", workerId, ++count);
                if (processed % steps == 0 || processed == total) {
                    logger.info("Completed {} requests", processed);
                }
            }
            logger.debug("Worker-{}: finished.", workerId);
            return result;
        }

        public void warmup() throws TranslateException {
            predictor.predict(null);
        }

        public void close() {
            predictor.close();
        }
    }
}
