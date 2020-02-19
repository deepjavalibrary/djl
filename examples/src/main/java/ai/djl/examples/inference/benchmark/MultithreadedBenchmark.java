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

import ai.djl.examples.inference.benchmark.util.AbstractBenchmark;
import ai.djl.examples.inference.benchmark.util.Arguments;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.listener.MemoryTrainingListener;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MultithreadedBenchmark extends AbstractBenchmark<BufferedImage, Classifications> {

    private static final Logger logger = LoggerFactory.getLogger(MultithreadedBenchmark.class);

    BufferedImage img;
    int numOfThreads;
    AtomicInteger callableNumber;
    AtomicInteger successThreads;
    ExecutorService executorService;

    public MultithreadedBenchmark() {
        super(BufferedImage.class, Classifications.class);
    }

    public static void main(String[] args) {
        if (new MultithreadedBenchmark().runBenchmark(args)) {
            System.exit(0); // NOPMD
        }
        System.exit(-1); // NOPMD
    }

    /** {@inheritDoc} */
    @Override
    protected void initialize(
            ZooModel<BufferedImage, Classifications> model, Arguments arguments, Metrics metrics)
            throws IOException {
        Path imageFile = arguments.getImageFile();
        img = BufferedImageUtils.fromFile(imageFile);

        numOfThreads = arguments.getThreads();
        callableNumber = new AtomicInteger();
        successThreads = new AtomicInteger();
        logger.info("Multithreaded inference with {} threads.", numOfThreads);
        metrics.addMetric("thread", numOfThreads);
        executorService = Executors.newFixedThreadPool(numOfThreads);
    }

    /** {@inheritDoc} */
    @Override
    protected CompletableFuture<Classifications> predict(
            ZooModel<BufferedImage, Classifications> model, Arguments arguments, Metrics metrics) {
        PredictorSupplier supplier = new PredictorSupplier(model, metrics);
        return CompletableFuture.supplyAsync(supplier, executorService);
    }

    /** {@inheritDoc} */
    @Override
    protected void clean() {
        executorService.shutdown();
        if (successThreads.get() != callableNumber.get()) {
            logger.error(
                    "Only {}/{} threads finished.", successThreads.get(), callableNumber.get());
        }
    }

    /** {@inheritDoc} */
    @Override
    protected Options getOptions() {
        Options options = super.getOptions();
        options.addOption(
                Option.builder("t")
                        .longOpt("threads")
                        .hasArg()
                        .argName("NUMBER_THREADS")
                        .desc("Number of inference threads.")
                        .build());
        return options;
    }

    private class PredictorSupplier implements Supplier<Classifications> {

        private Predictor<BufferedImage, Classifications> predictor;
        private Metrics metrics;
        private String workerId;
        private boolean collectMemory;

        public PredictorSupplier(ZooModel<BufferedImage, Classifications> model, Metrics metrics) {
            this.predictor = model.newPredictor();
            this.metrics = metrics;
            int iteration = callableNumber.getAndIncrement();
            this.workerId = String.format("%02d", iteration);
            this.collectMemory = iteration == 0;
            predictor.setMetrics(metrics);
        }

        /** {@inheritDoc} */
        @Override
        public Classifications get() {
            try {
                Classifications result = predictor.predict(img);
                if (collectMemory) {
                    MemoryTrainingListener.collectMemoryInfo(metrics);
                }
                logger.debug("Worker-{}: finished.", workerId);
                predictor.close();
                successThreads.incrementAndGet();
                return result;
            } catch (Exception e) {
                logger.error("Failed to classify with worker " + workerId, e);
                return null;
            }
        }
    }
}
