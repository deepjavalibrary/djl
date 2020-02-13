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
package ai.djl.examples.inference.benchmark.util;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.examples.inference.benchmark.MultithreadedBenchmark;
import ai.djl.metric.Metrics;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.listener.MemoryTrainingListener;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Abstract class that encapsulate command line options for example project. */
public abstract class AbstractBenchmark<I, O> {

    private static final Logger logger = LoggerFactory.getLogger(AbstractBenchmark.class);

    private Class<I> input;
    private Class<O> output;
    private O lastResult;

    protected ProgressBar progressBar;

    protected int maxIterations;
    protected int iterationCount;

    public AbstractBenchmark(Class<I> input, Class<O> output) {
        this.input = input;
        this.output = output;
    }

    /**
     * Returns last predict result.
     *
     * <p>This method is used for unit test only.
     *
     * @return last predict result
     */
    public O getPredictResult() {
        return lastResult;
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
            maxIterations = arguments.getIteration();
            if (this instanceof MultithreadedBenchmark) {
                maxIterations = Math.max(maxIterations, arguments.getThreads() * 2);
            }
            Duration duration = Duration.ofMinutes(arguments.getDuration());
            if (runByIterations()) {
                logger.info(
                        "Running {} on: {}, iterations: {}.",
                        getClass().getSimpleName(),
                        Device.defaultDevice(),
                        maxIterations);
                progressBar = new ProgressBar("Iteration", maxIterations);
            } else {
                logger.info(
                        "Running {} on: {}, duration: {} minutes.",
                        getClass().getSimpleName(),
                        Device.defaultDevice(),
                        duration.toMinutes());
                progressBar = new ProgressBar("Iteration", duration.getSeconds() * 1000);
            }
            Metrics metrics = new Metrics();
            long begin = System.currentTimeMillis();

            List<CompletableFuture<O>> predictResults = new ArrayList<>();
            long totalTime;
            try (ZooModel<I, O> model = loadModel(arguments, metrics, input, output)) {
                initialize(model, arguments, metrics);
                while (keepPredicting(duration, begin)) {
                    iterationCount++;
                    predictResults.add(predict(model, arguments, metrics));
                    updateProgress(progressBar, begin);
                }
                for (CompletableFuture<O> predictResult : predictResults) {
                    lastResult = predictResult.get();
                }
                totalTime = System.currentTimeMillis() - begin;

            } catch (Exception e) {
                logger.error("Failed to run benchmark", e);
                throw e;
            } finally {
                clean();
            }

            recordResults(arguments, metrics, totalTime);
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

    protected abstract void initialize(ZooModel<I, O> model, Arguments arguments, Metrics metrics)
            throws IOException;

    /**
     * Abstract predict method that must be implemented by sub class.
     *
     * @param model the model to use for predicting
     * @param arguments command line arguments
     * @param metrics {@link Metrics} to collect statistic information
     * @return prediction result
     * @throws TranslateException if error occurs when processing input or output
     */
    protected abstract CompletableFuture<O> predict(
            ZooModel<I, O> model, Arguments arguments, Metrics metrics) throws TranslateException;

    protected abstract void clean();

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

    protected ZooModel<I, O> loadModel(
            Arguments arguments, Metrics metrics, Class<I> input, Class<O> output)
            throws ModelException, IOException {
        long begin = System.nanoTime();

        Criteria.Builder<I, O> builder =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(input, output)
                        .optOptions(arguments.getCriteria())
                        .optProgress(new ProgressBar());

        String modelName = arguments.getModelName();
        if (modelName == null) {
            modelName = "resnet";
        }
        builder.optModelLoaderName(modelName);

        ZooModel<I, O> model = ModelZoo.loadModel(builder.build());
        long delta = System.nanoTime() - begin;
        logger.info(
                "Model {} loaded in: {} ms.",
                model.getName(),
                String.format("%.3f", delta / 1_000_000f));
        metrics.addMetric("LoadModel", delta);
        return model;
    }

    private boolean runByIterations() {
        return maxIterations != -1;
    }

    private boolean keepPredicting(Duration duration, long startTime) {
        if (runByIterations()) {
            return iterationCount < maxIterations;
        } else {
            return System.currentTimeMillis() - startTime < duration.getSeconds() * 1000;
        }
    }

    private void updateProgress(ProgressBar progressBar, long startTime) {
        if (runByIterations()) {
            progressBar.update(iterationCount);
        } else {
            progressBar.update(System.currentTimeMillis() - startTime);
        }
    }

    private void recordResults(Arguments arguments, Metrics metrics, long totalTime) {
        logger.info("Last inference result: {}", lastResult);
        logger.info(
                String.format(
                        "total time: %d ms, total runs: %d iterations", totalTime, iterationCount));

        if (metrics.hasMetric("LoadModel")) {
            long loadModelTime = metrics.getMetric("LoadModel").get(0).getValue().longValue();
            logger.info(
                    "Model loading time: {} ms.",
                    String.format("%.3f", loadModelTime / 1_000_000f));
        }

        if (metrics.hasMetric("Inference") && maxIterations > 1) {
            float p50 = metrics.percentile("Inference", 50).getValue().longValue() / 1_000_000f;
            float p90 = metrics.percentile("Inference", 90).getValue().longValue() / 1_000_000f;
            float p99 = metrics.percentile("Inference", 99).getValue().longValue() / 1_000_000f;
            float preP50 = metrics.percentile("Preprocess", 50).getValue().longValue() / 1_000_000f;
            float preP90 = metrics.percentile("Preprocess", 90).getValue().longValue() / 1_000_000f;
            float preP99 = metrics.percentile("Preprocess", 99).getValue().longValue() / 1_000_000f;
            float postP50 =
                    metrics.percentile("Postprocess", 50).getValue().longValue() / 1_000_000f;
            float postP90 =
                    metrics.percentile("Postprocess", 90).getValue().longValue() / 1_000_000f;
            float postP99 =
                    metrics.percentile("Postprocess", 99).getValue().longValue() / 1_000_000f;
            logger.info(
                    String.format(
                            "inference P50: %.3f ms, P90: %.3f ms, P99: %.3f ms", p50, p90, p99));
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
            MemoryTrainingListener.dumpMemoryInfo(metrics, arguments.getOutputDir());
        }
    }
}
