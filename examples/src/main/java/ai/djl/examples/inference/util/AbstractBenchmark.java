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
package ai.djl.examples.inference.util;

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.examples.util.MemoryUtils;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;
import ai.djl.zoo.ModelZoo;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.time.Duration;
import java.util.Map;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Abstract class that encapsulate command line options for example project. */
public abstract class AbstractBenchmark<T> {

    private static final Logger logger = LoggerFactory.getLogger(AbstractBenchmark.class);

    private T lastResult;

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
     */
    protected abstract T predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelException, TranslateException;

    /**
     * Returns command line options.
     *
     * <p>Child class can override this method and return different command line options.
     *
     * @return command line options
     */
    protected Options getOptions() {
        return Arguments.getOptions();
    }

    /**
     * Parse command line into arguments.
     *
     * <p>Child class can override this method and return extension of {@link Arguments}.
     *
     * @param cmd list of arguments parsed against a {@link Options} descriptor
     * @return parsed arguments
     */
    protected Arguments parseArguments(CommandLine cmd) {
        return new Arguments(cmd);
    }

    /**
     * Execute example code.
     *
     * @param args input raw arguments
     * @return if example execution complete successfully
     */
    public final boolean runBenchmark(String[] args) {
        Options options = getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            Arguments arguments = parseArguments(cmd);

            long init = System.nanoTime();
            String version = Engine.getInstance().getVersion();
            long loaded = System.nanoTime();
            logger.info(
                    String.format(
                            "Load library %s in %.3f ms.", version, (loaded - init) / 1_000_000f));
            Duration duration = Duration.ofMinutes(arguments.getDuration());
            if (arguments.getDuration() != 0) {
                logger.info(
                        "Running {} on: {}, duration: {} minutes.",
                        getClass().getSimpleName(),
                        Device.defaultDevice(),
                        duration.toMinutes());
            }
            int iteration = arguments.getIteration();

            while (!duration.isNegative()) {
                Metrics metrics = new Metrics(); // Reset Metrics for each test loop.
                logger.info(
                        "Running {} on: {}, iteration: {}.",
                        getClass().getSimpleName(),
                        Device.defaultDevice(),
                        iteration);
                progressBar = new ProgressBar("Iteration", iteration);
                long begin = System.currentTimeMillis();
                lastResult = predict(arguments, metrics, iteration);
                long totalTime = System.currentTimeMillis() - begin;

                logger.info("Inference result: {}", lastResult);
                int totalRuns = iteration;
                if (metrics.hasMetric("thread")) {
                    totalRuns *= metrics.getMetric("thread").get(0).getValue().intValue();
                }
                logger.info(
                        String.format(
                                "total time: %d ms, total runs: %d iterations",
                                totalTime, totalRuns));

                if (metrics.hasMetric("LoadModel")) {
                    long loadModelTime =
                            metrics.getMetric("LoadModel").get(0).getValue().longValue();
                    logger.info(
                            "Model loading time: {} ms.",
                            String.format("%.3f", loadModelTime / 1_000_000f));
                }

                if (metrics.hasMetric("Inference") && iteration > 1) {
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
                        float heap = metrics.percentile("Heap", 90).getValue().longValue();
                        float nonHeap = metrics.percentile("NonHeap", 90).getValue().longValue();
                        float cpu = metrics.percentile("cpu", 90).getValue().longValue();
                        float rss = metrics.percentile("rss", 90).getValue().longValue();

                        logger.info(String.format("heap P90: %.3f", heap));
                        logger.info(String.format("nonHeap P90: %.3f", nonHeap));
                        logger.info(String.format("cpu P90: %.3f", cpu));
                        logger.info(String.format("rss P90: %.3f", rss));
                    }
                }
                MemoryUtils.dumpMemoryInfo(metrics, arguments.getOutputDir());
                long delta = System.currentTimeMillis() - begin;
                duration = duration.minus(Duration.ofMillis(delta));
                if (!duration.isNegative()) {
                    logger.info(duration.toMinutes() + " minutes left");
                }
            }
            return true;
        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setLeftPadding(1);
            formatter.setWidth(120);
            formatter.printHelp(e.getMessage(), options);
        } catch (Throwable t) {
            logger.error("Unexpected error", t);
        }
        return false;
    }

    /**
     * Returns last predict result.
     *
     * <p>This method is used for unit test only.
     *
     * @return last predict result
     */
    public T getPredictResult() {
        return lastResult;
    }

    protected ZooModel<BufferedImage, Classifications> loadModel(
            Arguments arguments, Metrics metrics) throws ModelException, IOException {
        long begin = System.nanoTime();

        String modelName = arguments.getModelName();
        if (modelName == null) {
            modelName = "RESNET";
        }

        Map<String, String> criteria = arguments.getCriteria();
        ModelLoader<BufferedImage, Classifications> loader;
        if (arguments.isImperative()) {
            loader = ModelZoo.getModelLoader(modelName);
        } else {
            loader = MxModelZoo.getModelLoader(modelName);
        }

        Progress progress = new ProgressBar();
        ZooModel<BufferedImage, Classifications> model = loader.loadModel(criteria, progress);
        long delta = System.nanoTime() - begin;
        logger.info(
                "Model {} loaded in: {} ms.",
                model.getName(),
                String.format("%.3f", delta / 1_000_000f));
        metrics.addMetric("LoadModel", delta);
        return model;
    }
}
