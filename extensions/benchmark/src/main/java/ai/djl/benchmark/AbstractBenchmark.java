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
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.listener.MemoryTrainingListener;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.time.Duration;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Abstract benchmark class. */
public abstract class AbstractBenchmark {

    private static final Logger logger = LoggerFactory.getLogger(AbstractBenchmark.class);

    protected ProgressBar progressBar;

    /**
     * Abstract predict method that must be implemented by sub class.
     *
     * @param arguments command line arguments
     * @param metrics {@link Metrics} to collect statistic information
     * @param iteration number of prediction iteration to run
     * @return prediction result
     * @throws IOException if io error occurs when loading model.
     * @throws ModelException if specified model not found or there is a parameter error
     * @throws TranslateException if error occurs when processing input or output
     * @throws ClassNotFoundException if input or output class cannot be loaded
     */
    protected abstract float[] predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelException, TranslateException, ClassNotFoundException;

    /**
     * Execute benchmark.
     *
     * @param args input raw arguments
     * @return if example execution complete successfully
     */
    public final boolean runBenchmark(String[] args) {
        Options options = Arguments.getOptions();
        try {
            if (Arguments.hasHelp(args)) {
                Arguments.printHelp(
                        "usage: djl-bench [-p MODEL-PATH] -s INPUT-SHAPES [OPTIONS]", options);
                return true;
            }
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            Arguments arguments = new Arguments(cmd);
            String engineName = arguments.getEngine();
            Engine engine = Engine.getEngine(engineName);

            long init = System.nanoTime();
            String version = engine.getVersion();
            long loaded = System.nanoTime();
            logger.info(
                    String.format(
                            "Load %s (%s) in %.3f ms.",
                            engineName, version, (loaded - init) / 1_000_000f));
            Duration duration = Duration.ofSeconds(arguments.getDuration());
            Object devices;
            if (this instanceof MultithreadedBenchmark) {
                devices = engine.getDevices(arguments.getMaxGpus());
            } else {
                devices = engine.defaultDevice();
            }

            if (arguments.getDuration() != 0) {
                logger.info(
                        "Running {} on: {}, duration: {} minutes.",
                        getClass().getSimpleName(),
                        devices,
                        duration.toMinutes());
            } else {
                logger.info("Running {} on: {}.", getClass().getSimpleName(), devices);
            }
            int numOfThreads = arguments.getThreads();
            int iteration = arguments.getIteration();
            if (this instanceof MultithreadedBenchmark) {
                int expected = 10 * numOfThreads;
                if (iteration < expected) {
                    iteration = expected;
                    logger.info(
                            "Iteration is too small for multi-threading benchmark. Adjust to: {}",
                            iteration);
                }
            }
            while (!duration.isNegative()) {
                Metrics metrics = new Metrics(); // Reset Metrics for each test loop.
                progressBar = new ProgressBar("Iteration", iteration);
                float[] lastResult = predict(arguments, metrics, iteration);
                if (lastResult == null) {
                    return false;
                }

                long begin = metrics.getMetric("start").get(0).getValue().longValue();
                long end = metrics.getMetric("end").get(0).getValue().longValue();
                long totalTime = end - begin;

                if (lastResult.length > 3) {
                    logger.info(
                            "Inference result: [{}, {}, {} ...]",
                            lastResult[0],
                            lastResult[1],
                            lastResult[2]);
                } else {
                    logger.info("Inference result: {}", lastResult);
                }

                String throughput = String.format("%.2f", iteration * 1000d / totalTime);
                logger.info(
                        "Throughput: {}, completed {} iteration in {} ms.",
                        throughput,
                        iteration,
                        totalTime);

                if (metrics.hasMetric("LoadModel")) {
                    long loadModelTime =
                            metrics.getMetric("LoadModel").get(0).getValue().longValue();
                    logger.info(
                            "Model loading time: {} ms.",
                            String.format("%.3f", loadModelTime / 1_000_000f));
                }

                if (metrics.hasMetric("Inference") && iteration > 1) {
                    float totalP50 =
                            metrics.percentile("Total", 50).getValue().longValue() / 1_000_000f;
                    float totalP90 =
                            metrics.percentile("Total", 90).getValue().longValue() / 1_000_000f;
                    float totalP99 =
                            metrics.percentile("Total", 99).getValue().longValue() / 1_000_000f;
                    float p50 =
                            metrics.percentile("Inference", 50).getValue().longValue() / 1_000_000f;
                    float p90 =
                            metrics.percentile("Inference", 90).getValue().longValue() / 1_000_000f;
                    float p99 =
                            metrics.percentile("Inference", 99).getValue().longValue() / 1_000_000f;
                    float preP50 =
                            metrics.percentile("Preprocess", 50).getValue().longValue()
                                    / 1_000_000f;
                    float preP90 =
                            metrics.percentile("Preprocess", 90).getValue().longValue()
                                    / 1_000_000f;
                    float preP99 =
                            metrics.percentile("Preprocess", 99).getValue().longValue()
                                    / 1_000_000f;
                    float postP50 =
                            metrics.percentile("Postprocess", 50).getValue().longValue()
                                    / 1_000_000f;
                    float postP90 =
                            metrics.percentile("Postprocess", 90).getValue().longValue()
                                    / 1_000_000f;
                    float postP99 =
                            metrics.percentile("Postprocess", 99).getValue().longValue()
                                    / 1_000_000f;
                    logger.info(
                            String.format(
                                    "total P50: %.3f ms, P90: %.3f ms, P99: %.3f ms",
                                    totalP50, totalP90, totalP99));
                    logger.info(
                            String.format(
                                    "inference P50: %.3f ms, P90: %.3f ms, P99: %.3f ms",
                                    p50, p90, p99));
                    logger.info(
                            String.format(
                                    "preprocess P50: %.3f ms, P90: %.3f ms, P99: %.3f ms",
                                    preP50, preP90, preP99));
                    logger.info(
                            String.format(
                                    "postprocess P50: %.3f ms, P90: %.3f ms, P99: %.3f ms",
                                    postP50, postP90, postP99));

                    if (Boolean.getBoolean("collect-memory")) {
                        float heapBeforeModel =
                                metrics.getMetric("Heap").get(0).getValue().longValue();
                        float heapBeforeInference =
                                metrics.getMetric("Heap").get(1).getValue().longValue();
                        float heap = metrics.percentile("Heap", 90).getValue().longValue();
                        float nonHeap = metrics.percentile("NonHeap", 90).getValue().longValue();
                        int mb = 1024 * 1024;
                        logger.info(String.format("heap (base): %.3f MB", heapBeforeModel / mb));
                        logger.info(
                                String.format("heap (model): %.3f MB", heapBeforeInference / mb));
                        logger.info(String.format("heap P90: %.3f MB", heap / mb));
                        logger.info(String.format("nonHeap P90: %.3f MB", nonHeap / mb));

                        if (!System.getProperty("os.name").startsWith("Win")) {
                            float rssBeforeModel =
                                    metrics.getMetric("rss").get(0).getValue().longValue();
                            float rssBeforeInference =
                                    metrics.getMetric("rss").get(1).getValue().longValue();
                            float rss = metrics.percentile("rss", 90).getValue().longValue();
                            float cpu = metrics.percentile("cpu", 90).getValue().longValue();
                            logger.info(String.format("cpu P90: %.3f %%", cpu));
                            logger.info(String.format("rss (base): %.3f MB", rssBeforeModel / mb));
                            logger.info(
                                    String.format("rss (model): %.3f MB", rssBeforeInference / mb));
                            logger.info(String.format("rss P90: %.3f MB", rss / mb));
                        }
                    }
                }
                MemoryTrainingListener.dumpMemoryInfo(metrics, arguments.getOutputDir());
                long delta = System.currentTimeMillis() - begin;
                duration = duration.minus(Duration.ofMillis(delta));
                if (!duration.isNegative()) {
                    logger.info(duration.toMinutes() + " minutes left");
                }
            }
            return true;
        } catch (ParseException e) {
            Arguments.printHelp(e.getMessage(), options);
        } catch (Throwable t) {
            logger.error("Unexpected error", t);
        }
        return false;
    }

    protected ZooModel<Void, float[]> loadModel(Arguments arguments, Metrics metrics, Device device)
            throws ModelException, IOException {
        long begin = System.nanoTime();
        PairList<DataType, Shape> shapes = arguments.getInputShapes();
        BenchmarkTranslator translator = new BenchmarkTranslator(shapes);

        Criteria<Void, float[]> criteria =
                Criteria.builder()
                        .setTypes(Void.class, float[].class)
                        .optModelUrls(arguments.getModelUrl())
                        .optModelName(arguments.getModelName())
                        .optEngine(arguments.getEngine())
                        .optOptions(arguments.getModelOptions())
                        .optArguments(arguments.getModelArguments())
                        .optDevice(device)
                        .optTranslator(translator)
                        .optProgress(new ProgressBar())
                        .build();

        ZooModel<Void, float[]> model = criteria.loadModel();
        if (device == Device.cpu() || device == Device.gpu()) {
            long delta = System.nanoTime() - begin;
            logger.info(
                    "Model {} loaded in: {} ms.",
                    model.getName(),
                    String.format("%.3f", delta / 1_000_000f));
            metrics.addMetric("LoadModel", delta);
        }
        return model;
    }

    private static final class BenchmarkTranslator implements NoBatchifyTranslator<Void, float[]> {

        private PairList<DataType, Shape> shapes;

        public BenchmarkTranslator(PairList<DataType, Shape> shapes) {
            this.shapes = shapes;
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Void input) {
            NDList list = new NDList();
            for (Pair<DataType, Shape> pair : shapes) {
                DataType dataType = pair.getKey();
                Shape shape = pair.getValue();
                list.add(ctx.getNDManager().zeros(shape, dataType));
            }
            return list;
        }

        /** {@inheritDoc} */
        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            FloatBuffer fb = list.get(0).toByteBuffer().asFloatBuffer();
            float[] ret = new float[fb.remaining()];
            fb.get(ret);
            return ret;
        }
    }
}
