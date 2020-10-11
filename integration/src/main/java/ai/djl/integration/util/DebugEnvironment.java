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

import ai.djl.engine.Engine;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

/** Emits debug information about the user environment. */
@SuppressWarnings("PMD.SystemPrintln")
public final class DebugEnvironment {

    private DebugEnvironment() {}

    /**
     * Runs the debug environment script.
     *
     * @param args program arguments
     * @throws IOException if failed to get environment data
     */
    public static void main(String[] args) throws IOException {
        Engine.debugEnvironment();

        System.out.println();
        System.out.println("--------------- Hardware --------------");
        hardware();
    }

    // Based on https://stackoverflow.com/a/25596
    private static void hardware() throws IOException {
        Runtime rt = Runtime.getRuntime();

        /* Total number of processors or cores available to the JVM */
        System.out.println("Available processors (cores): " + rt.availableProcessors());

        System.out.println("Byte Order: " + ByteOrder.nativeOrder().toString());

        /* Total amount of free memory available to the JVM */
        System.out.println("Free memory (bytes): " + rt.freeMemory());

        /* This will return Long.MAX_VALUE if there is no preset limit */
        long maxMemory = rt.maxMemory();
        /* Maximum amount of memory the JVM will attempt to use */
        System.out.println(
                "Maximum memory (bytes): "
                        + (maxMemory == Long.MAX_VALUE ? "no limit" : maxMemory));

        /* Total memory currently available to the JVM */
        System.out.println("Total memory available to JVM (bytes): " + rt.totalMemory());

        MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();
        MemoryUsage heap = memBean.getHeapMemoryUsage();
        MemoryUsage nonHeap = memBean.getNonHeapMemoryUsage();

        System.out.println("Heap committed: " + heap.getCommitted());
        System.out.println("Heap nonCommitted: " + nonHeap.getCommitted());

        if (!TestUtils.isWindows()) {
            System.out.println("GCC: ");
            Process process = rt.exec("gcc --version");
            try (Scanner gccOut =
                    new Scanner(process.getInputStream(), StandardCharsets.UTF_8.name())) {
                gccOut.useDelimiter(System.lineSeparator());
                while (gccOut.hasNext()) {
                    System.out.println(gccOut.next());
                }
            }
        }
    }
}
