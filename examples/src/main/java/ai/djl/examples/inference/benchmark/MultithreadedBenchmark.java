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
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.listener.MemoryTrainingListener;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
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
    public Classifications predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelException {
        Path imageFile = arguments.getImageFile();
        BufferedImage img = BufferedImageUtils.fromFile(imageFile);

        ZooModel<BufferedImage, Classifications> model = loadModel(arguments, metrics);

        int numOfThreads = arguments.getThreads();
        AtomicInteger counter = new AtomicInteger(iteration);
        logger.info("Multithreaded inference with {} threads.", numOfThreads);

        List<PredictorCallable> callables = new ArrayList<>(numOfThreads);
        for (int i = 0; i < numOfThreads; ++i) {
            callables.add(new PredictorCallable(model, img, metrics, counter, i, i == 0));
        }

        Classifications classification = null;
        ExecutorService executorService = Executors.newFixedThreadPool(numOfThreads);
        int successThreads = 0;
        try {
            List<Future<Classifications>> futures = executorService.invokeAll(callables);
            for (Future<Classifications> future : futures) {
                try {
                    classification = future.get();
                    ++successThreads;
                } catch (InterruptedException | ExecutionException e) {
                    logger.error("", e);
                }
            }
        } catch (InterruptedException e) {
            logger.error("", e);
        } finally {
            executorService.shutdown();
        }
        if (successThreads != numOfThreads) {
            logger.error("Only {}/{} threads finished.", successThreads, numOfThreads);
        }

        return classification;
    }

    private static class PredictorCallable implements Callable<Classifications> {

        private Predictor<BufferedImage, Classifications> predictor;
        private BufferedImage img;
        private Metrics metrics;
        private String workerId;
        private boolean collectMemory;
        private AtomicInteger counter;

        public PredictorCallable(
                ZooModel<BufferedImage, Classifications> model,
                BufferedImage img,
                Metrics metrics,
                AtomicInteger counter,
                int workerId,
                boolean collectMemory) {
            this.predictor = model.newPredictor();
            this.img = img;
            this.metrics = metrics;
            this.counter = counter;
            this.workerId = String.format("%02d", workerId);
            this.collectMemory = collectMemory;
            predictor.setMetrics(metrics);
        }

        /** {@inheritDoc} */
        @Override
        public Classifications call() throws TranslateException {
            Classifications result = null;
            int count = 0;
            while (counter.decrementAndGet() > 0) {
                result = predictor.predict(img);
                if (collectMemory) {
                    MemoryTrainingListener.collectMemoryInfo(metrics);
                }
                logger.trace("Worker-{}: {} iteration finished.", workerId, ++count);
            }
            logger.debug("Worker-{}: finished.", workerId);
            return result;
        }
    }
}
