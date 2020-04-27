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
package ai.djl.integration.util;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.util.cuda.CudaUtils;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Emits debug information about the user environment. */
public final class DebugEnvironment {

    private static final Logger logger = LoggerFactory.getLogger(DebugEnvironment.class);

    private DebugEnvironment() {}

    /**
     * Runs the debug environment script.
     *
     * @param args program arguments
     * @throws IOException if failed to get environment data
     */
    public static void main(String[] args) throws IOException {
        logger.info("----------System Properties----------");
        System.getProperties().forEach((k, v) -> logger.info(k + ": " + v));

        logger.info("");
        logger.info("----------Environment Variables----------");
        System.getenv().forEach((k, v) -> logger.info(k + ": " + v));

        logger.info("");
        logger.info("----------Default Engine----------");
        Engine engine = Engine.getInstance();
        engine.debugEnvironment();

        logger.info("");
        logger.info("----------Hardware----------");
        hardware();
    }

    // Based on https://stackoverflow.com/a/25596
    private static void hardware() throws IOException {
        Runtime rt = Runtime.getRuntime();

        /* Total number of processors or cores available to the JVM */
        logger.info("Available processors (cores): {}", rt.availableProcessors());

        logger.info("Byte Order: " + ByteOrder.nativeOrder().toString());

        /* Total amount of free memory available to the JVM */
        logger.info("Free memory (bytes): {}", rt.freeMemory());

        /* This will return Long.MAX_VALUE if there is no preset limit */
        long maxMemory = rt.maxMemory();
        /* Maximum amount of memory the JVM will attempt to use */
        logger.info(
                "Maximum memory (bytes): {}",
                (maxMemory == Long.MAX_VALUE ? "no limit" : maxMemory));

        /* Total memory currently available to the JVM */
        logger.info("Total memory available to JVM (bytes): {}", rt.totalMemory());

        MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();
        MemoryUsage heap = memBean.getHeapMemoryUsage();
        MemoryUsage nonHeap = memBean.getNonHeapMemoryUsage();

        logger.info("Heap committed: {}", heap.getCommitted());
        logger.info("Heap nonCommitted: {}", nonHeap.getCommitted());

        int gpuCount = Device.getGpuCount();
        logger.info("GPU Count: {}", gpuCount);
        logger.info("Default Device: {}", Device.defaultDevice());

        // CudaUtils.getGpuMemory() will allocates memory on GPUs if CUDA runtime is not
        // initialized.
        for (int i = 0; i < gpuCount; ++i) {
            Device device = Device.gpu(i);
            MemoryUsage mem = CudaUtils.getGpuMemory(device);
            logger.info("GPU {} memory used: {} bytes", i, mem.getCommitted());
        }

        if (!TestUtils.isWindows()) {
            logger.info("GCC: ");
            Process process = rt.exec("gcc --version");
            try (Scanner gccOut =
                    new Scanner(process.getInputStream(), StandardCharsets.UTF_8.name())) {
                gccOut.useDelimiter(System.lineSeparator());
                while (gccOut.hasNext()) {
                    logger.info(gccOut.next());
                }
            }
        }
    }
}
