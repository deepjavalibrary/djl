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
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.time.Duration;
import java.util.List;
import java.util.ListIterator;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Abstract class that encapsulate command line options for example project. */
public abstract class AbstractInference<T> {

    private static final Logger logger = LoggerFactory.getLogger(AbstractInference.class);

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
    public final boolean runExample(String[] args) {
        Options options = getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            Arguments arguments = parseArguments(cmd);

            Duration duration = Duration.ofMinutes(arguments.getDuration());
            int iteration = arguments.getIteration();
            progressBar = new ProgressBar("Iteration", iteration);

            logger.info(
                    "Running {} on: {}, iteration: {}, duration: {} minutes.",
                    getClass().getSimpleName(),
                    Device.defaultDevice(),
                    iteration,
                    duration.toMinutes());

            long init = System.nanoTime();
            String version = Engine.getInstance().getVersion();
            long loaded = System.nanoTime();
            logger.info(
                    String.format(
                            "Load library %s in %.3f ms.", version, (loaded - init) / 1_000_000f));

            while (!duration.isNegative()) {
                Metrics metrics = new Metrics(); // Reset Metrics for each test loop.

                long begin = System.currentTimeMillis();
                lastResult = predict(arguments, metrics, iteration);

                logger.info("Inference result: {}", lastResult);

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
                }

                MemoryUtils.dumpMemoryInfo(metrics, arguments.getLogDir());
                long delta = System.currentTimeMillis() - begin;
                duration = duration.minus(Duration.ofMillis(delta));
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
     * Load MXNet synset.txt file into array of string.
     *
     * @param inputStream sysnet.txt input
     * @return array of string
     */
    public static String[] loadSynset(InputStream inputStream) {
        List<String> output = Utils.readLines(inputStream);
        ListIterator<String> it = output.listIterator();
        while (it.hasNext()) {
            String synsetLemma = it.next();
            it.set(synsetLemma.substring(synsetLemma.indexOf(' ') + 1));
        }
        return output.toArray(new String[0]);
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
}
