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
package software.amazon.ai.examples.inference.util;

import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.lang.management.RuntimeMXBean;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.mxnet.zoo.ModelNotFoundException;
import org.slf4j.Logger;
import software.amazon.ai.Device;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.metric.Metric;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.translate.TranslateException;
import software.amazon.ai.util.Utils;

/** Abstract class that encapsulate command line options for example project. */
public abstract class AbstractExample {

    private static final Logger logger = LogUtils.getLogger(AbstractExample.class);

    private static Object lastResult;

    /**
     * Abstract predict method that must be implemented by sub class.
     *
     * @param arguments command line arguments
     * @param metrics {@link Metrics} to collect statistic information
     * @param iteration number of prediction iteration to run
     * @return prediction result
     * @throws IOException if io error occurs when loading model.
     * @throws ModelNotFoundException if specified model not found
     * @throws TranslateException if error occurs when processing input or output
     */
    protected abstract Object predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelNotFoundException, TranslateException;

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
                setLastResult(predict(arguments, metrics, iteration));

                logger.info("Inference result: {}", lastResult);

                float p50 = metrics.percentile("Inference", 50).getValue().longValue() / 1_000_000f;
                float p90 = metrics.percentile("Inference", 90).getValue().longValue() / 1_000_000f;

                logger.info(String.format("inference P50: %.3f ms, P90: %.3f ms", p50, p90));

                dumpMemoryInfo(metrics, arguments.getLogDir());

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
     * Set predict result.
     *
     * <p>This method is used for unit test only.
     *
     * @param lastResult last predict result
     */
    private static void setLastResult(Object lastResult) {
        AbstractExample.lastResult = lastResult;
    }

    /**
     * Returns last predict result.
     *
     * <p>This method is used for unit test only.
     *
     * @return last predict result
     */
    public static Object getPredictResult() {
        return lastResult;
    }

    /**
     * Collect memory information.
     *
     * @param metrics {@link Metrics} to store memory information
     */
    protected void collectMemoryInfo(Metrics metrics) {
        MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();
        MemoryUsage heap = memBean.getHeapMemoryUsage();
        MemoryUsage nonHeap = memBean.getNonHeapMemoryUsage();

        long heapCommitted = heap.getCommitted();
        long nonHeapCommitted = nonHeap.getCommitted();
        getProcessInfo(metrics);

        metrics.addMetric("Heap", heapCommitted, "bytes");
        metrics.addMetric("NonHeap", nonHeapCommitted, "bytes");
        Engine engine = Engine.getInstance();
        int gpuCount = engine.getGpuCount();
        for (int i = 0; i < gpuCount; ++i) {
            Device device = Device.gpu(i);
            MemoryUsage mem = engine.getGpuMemory(device);
            metrics.addMetric("GPU-" + i, mem.getCommitted(), "bytes");
        }
    }

    /**
     * Dump memory metrics into log directory.
     *
     * @param metrics metrics contains memory information
     * @param logDir output log directory
     */
    protected void dumpMemoryInfo(Metrics metrics, String logDir) {
        if (logDir == null) {
            return;
        }

        try {
            Path dir = Paths.get(logDir);
            Files.createDirectories(dir);
            Path file = dir.resolve("memory.log");
            try (BufferedWriter writer =
                    Files.newBufferedWriter(
                            file, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
                List<Metric> list = new ArrayList<>();
                list.addAll(metrics.getMetric("Heap"));
                list.addAll(metrics.getMetric("NonHeap"));
                list.addAll(metrics.getMetric("cpu"));
                list.addAll(metrics.getMetric("rss"));
                int gpuCount = Engine.getInstance().getGpuCount();
                for (int i = 0; i < gpuCount; ++i) {
                    list.addAll(metrics.getMetric("GPU-" + i));
                }
                for (Metric metric : list) {
                    writer.append(metric.toString());
                    writer.newLine();
                }
            }
        } catch (IOException e) {
            logger.error("Failed dump memory log", e);
        }
    }

    /**
     * Print inference iteration progress.
     *
     * @param iteration total number of iteration
     * @param index index of iteration
     */
    @SuppressWarnings("PMD.SystemPrintln")
    protected void printProgress(int iteration, int index) {
        System.out.print(".");
        if (index % 80 == 79 || index == iteration - 1) {
            System.out.println();
        }
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

    private void getProcessInfo(Metrics metrics) {
        if (System.getProperty("os.name").startsWith("Linux")
                || System.getProperty("os.name").startsWith("Mac")) {
            // This solution only work for Linux like system.
            RuntimeMXBean mxBean = ManagementFactory.getRuntimeMXBean();
            String pid = mxBean.getName().split("@")[0];
            String cmd = "ps -o %cpu= -o rss= -p " + pid;
            try {
                Process process = Runtime.getRuntime().exec(cmd);
                try (InputStream is = process.getInputStream()) {
                    String line = new String(readAll(is), StandardCharsets.UTF_8).trim();
                    String[] tokens = line.split("\\s+");
                    if (tokens.length != 2) {
                        logger.error("Invalid ps output: " + line);
                        return;
                    }
                    float cpu = Float.parseFloat(tokens[0]);
                    long rss = Long.parseLong(tokens[1]);
                    metrics.addMetric("cpu", cpu, "%");
                    metrics.addMetric("rss", rss, "KB");
                }
            } catch (IOException e) {
                logger.error("Failed execute cmd: " + cmd, e);
            }
        }
    }

    private static byte[] readAll(InputStream is) throws IOException {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            int read;
            byte[] buf = new byte[8192];
            while ((read = is.read(buf)) != -1) {
                bos.write(buf, 0, read);
            }
            return bos.toByteArray();
        }
    }
}
