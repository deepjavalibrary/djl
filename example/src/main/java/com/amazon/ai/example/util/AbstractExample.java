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
package com.amazon.ai.example.util;

import com.amazon.ai.Context;
import com.amazon.ai.TranslateException;
import com.amazon.ai.engine.Engine;
import com.amazon.ai.metric.Metric;
import com.amazon.ai.metric.Metrics;
import com.sun.jna.Platform;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.lang.management.RuntimeMXBean;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
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
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;

public abstract class AbstractExample {

    private static final Logger logger = LogUtils.getLogger(AbstractExample.class);

    protected abstract void predict(Arguments arguments, int iteration)
            throws IOException, TranslateException;

    public void runExample(String[] args) {
        Options options = getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            Arguments arguments = parseArguments(cmd);

            Duration duration = Duration.ofMinutes(arguments.getDuration());
            int iteration = arguments.getIteration();

            logger.info("Running {}, iteration: {}", getClass().getSimpleName(), iteration);

            long init = System.nanoTime();
            String version = Engine.getInstance().getVersion();
            JnaUtils.getAllOpNames();
            long loaded = System.nanoTime();
            logger.info(
                    String.format(
                            "Load library %s in %.3f ms.", version, (loaded - init) / 1000000f));

            while (!duration.isNegative()) {
                long begin = System.currentTimeMillis();
                predict(arguments, iteration);
                long delta = System.currentTimeMillis() - begin;
                duration = duration.minus(Duration.ofMillis(delta));
            }
        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setLeftPadding(1);
            formatter.setWidth(120);
            formatter.printHelp(e.getMessage(), options);
        } catch (Throwable t) {
            logger.error("Unexpected error", t);
        }
    }

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
            Context context = Context.gpu(i);
            MemoryUsage mem = engine.getGpuMemory(context);
            metrics.addMetric("GPU-" + i, mem.getCommitted(), "bytes");
        }
    }

    protected void dumpMemoryInfo(Metrics metrics, String logDir) {
        if (logDir == null) {
            return;
        }

        try {
            File dir = new File(logDir);
            if (!dir.exists()) {
                FileUtils.forceMkdir(dir);
            }
            Path file = dir.toPath().resolve("memory.log");
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

    private void getProcessInfo(Metrics metrics) {
        if (Platform.isLinux() || Platform.isMac()) {
            // This solution only work for Linux like system.
            RuntimeMXBean mxBean = ManagementFactory.getRuntimeMXBean();
            String pid = mxBean.getName().split("@")[0];
            String cmd = "ps -o %cpu= -o rss= -p " + pid;
            try {
                Process process = Runtime.getRuntime().exec(cmd);
                try (InputStream is = process.getInputStream()) {
                    String line = IOUtils.toString(is, StandardCharsets.UTF_8).trim();
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

    protected Options getOptions() {
        return Arguments.getOptions();
    }

    protected Arguments parseArguments(CommandLine cmd) {
        return new Arguments(cmd);
    }

    @SuppressWarnings("PMD.SystemPrintln")
    protected void printProgress(int iteration, int index, String message) {
        if (index == 0) {
            logger.info(String.format("Result: %s", message));
        } else {
            System.out.print(".");
            if (index % 80 == 0 || index == iteration - 1) {
                System.out.println();
            }
        }
    }

    public static String[] loadSynset(InputStream inputStream) {
        try {
            List<String> output = IOUtils.readLines(inputStream, StandardCharsets.UTF_8);
            ListIterator<String> it = output.listIterator();
            while (it.hasNext()) {
                String synsetLemma = it.next();
                it.set(synsetLemma.substring(synsetLemma.indexOf(' ') + 1));
            }
            return output.toArray(JnaUtils.EMPTY_ARRAY);
        } catch (IOException e) {
            logger.warn("Error opening synset file.", e);
        }
        return JnaUtils.EMPTY_ARRAY;
    }
}
