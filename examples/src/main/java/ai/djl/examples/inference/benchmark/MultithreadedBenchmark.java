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
package ai.djl.examples.inference.benchmark;

import ai.djl.ModelException;
import ai.djl.examples.inference.benchmark.util.AbstractBenchmark;
import ai.djl.examples.inference.benchmark.util.Arguments;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.listener.MemoryTrainingListener;
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

public class MultithreadedBenchmark extends AbstractBenchmark {

    private static final Logger logger = LoggerFactory.getLogger(MultithreadedBenchmark.class);

    public static void main(String[] args) {
        if (new MultithreadedBenchmark().runBenchmark(args)) {
            System.exit(0); // NOPMD
        }
        System.exit(-1); // NOPMD
    }

    /** {@inheritDoc} */
    @Override
    public Object predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelException {
        Object inputData = arguments.getInputData();
        ZooModel<?, ?> model = loadModel(arguments, metrics);

        int numOfThreads = arguments.getThreads();
        int delay = arguments.getDelay();
        AtomicInteger counter = new AtomicInteger(iteration);
        logger.info("Multithreaded inference with {} threads.", numOfThreads);

        List<PredictorCallable> callables = new ArrayList<>(numOfThreads);
        for (int i = 0; i < numOfThreads; ++i) {
            callables.add(new PredictorCallable(model, inputData, metrics, counter, i, i == 0));
        }

        Object classification = null;
        ExecutorService executorService = Executors.newFixedThreadPool(numOfThreads);
        int successThreads = 0;
        try {
            metrics.addMetric("mt_start", System.currentTimeMillis(), "mills");
            try {
                List<Future<Object>> futures;
                if (delay > 0) {
                    futures = new ArrayList<>();
                    for (PredictorCallable callable : callables) {
                        futures.add(executorService.submit(callable));
                        Thread.sleep(delay);
                    }
                } else {
                    futures = executorService.invokeAll(callables);
                }

                for (Future<Object> future : futures) {
                    classification = future.get();
                    if (classification != null) {
                        ++successThreads;
                    }
                }
            } catch (InterruptedException | ExecutionException e) {
                logger.error("", e);
            }
            for (PredictorCallable callable : callables) {
                callable.close();
            }
        } finally {
            executorService.shutdown();
        }
        if (successThreads != numOfThreads) {
            logger.error("Only {}/{} threads finished.", successThreads, numOfThreads);
            return null;
        }

        return classification;
    }

    private static class PredictorCallable implements Callable<Object> {

        @SuppressWarnings("rawtypes")
        private Predictor predictor;

        private Object inputData;
        private Metrics metrics;
        private String workerId;
        private boolean collectMemory;
        private AtomicInteger counter;
        private int total;
        private int steps;

        public PredictorCallable(
                ZooModel<?, ?> model,
                Object inputData,
                Metrics metrics,
                AtomicInteger counter,
                int workerId,
                boolean collectMemory) {
            this.predictor = model.newPredictor();
            this.inputData = inputData;
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
        @SuppressWarnings("unchecked")
        public Object call() throws Exception {
            Object result = null;
            int count = 0;
            int remaining;
            while ((remaining = counter.decrementAndGet()) > 0 || result == null) {
                try {
                    result = predictor.predict(inputData);
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

        public void close() {
            predictor.close();
        }
    }
}
