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
package ai.djl.training.listener;

import ai.djl.Device;
import ai.djl.metric.Metric;
import ai.djl.metric.Metrics;
import ai.djl.training.Trainer;
import ai.djl.util.cuda.CudaUtils;
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
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link TrainingListener} that collects the memory usage information.
 *
 * <p>If an outputDir is provided, the file "$outputDir/memory.log" will be created after training
 * with the memory usage results.
 */
public class MemoryTrainingListener extends TrainingListenerAdapter {

    private static final Logger logger = LoggerFactory.getLogger(MemoryTrainingListener.class);

    private String outputDir;

    /** Constructs a {@link MemoryTrainingListener} that does not output data to a file. */
    public MemoryTrainingListener() {}

    /**
     * Constructs a {@link MemoryTrainingListener} that outputs data in the given directory.
     *
     * @param outputDir the directory to output the tracked memory data in
     */
    public MemoryTrainingListener(String outputDir) {
        this.outputDir = outputDir;
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {
        Metrics metrics = trainer.getMetrics();
        collectMemoryInfo(metrics);
    }

    /** {@inheritDoc} */
    @Override
    public void onValidationBatch(Trainer trainer, BatchData batchData) {
        Metrics metrics = trainer.getMetrics();
        collectMemoryInfo(metrics);
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingEnd(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        dumpMemoryInfo(metrics, outputDir);
    }

    /**
     * Collect memory information.
     *
     * @param metrics {@link Metrics} to store memory information
     */
    public static void collectMemoryInfo(Metrics metrics) {
        if (metrics != null && Boolean.getBoolean("collect-memory")) {
            MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();
            MemoryUsage heap = memBean.getHeapMemoryUsage();
            MemoryUsage nonHeap = memBean.getNonHeapMemoryUsage();

            long heapCommitted = heap.getCommitted();
            long nonHeapCommitted = nonHeap.getCommitted();
            getProcessInfo(metrics);

            metrics.addMetric("Heap", heapCommitted, "bytes");
            metrics.addMetric("NonHeap", nonHeapCommitted, "bytes");
            int gpuCount = Device.getGpuCount();

            // CudaUtils.getGpuMemory() will allocates memory on GPUs if CUDA runtime is not
            // initialized.
            for (int i = 0; i < gpuCount; ++i) {
                Device device = Device.gpu(i);
                MemoryUsage mem = CudaUtils.getGpuMemory(device);
                metrics.addMetric("GPU-" + i, mem.getCommitted(), "bytes");
            }
        }
    }

    /**
     * Dump memory metrics into log directory.
     *
     * @param metrics metrics contains memory information
     * @param logDir output log directory
     */
    public static void dumpMemoryInfo(Metrics metrics, String logDir) {
        if (metrics == null || logDir == null) {
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
                int gpuCount = Device.getGpuCount();
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

    private static void getProcessInfo(Metrics metrics) {
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
                        logger.error("Invalid ps output: {}", line);
                        return;
                    }
                    float cpu = Float.parseFloat(tokens[0]);
                    long rss = Long.parseLong(tokens[1]) * 1024;
                    metrics.addMetric("cpu", cpu, "%");
                    metrics.addMetric("rss", rss, "bytes");
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
