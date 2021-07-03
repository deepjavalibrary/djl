/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import org.testng.annotations.Test;

public class BenchmarkTest {

    @Test
    public void testHelp() {
        String[] args = {"-h"};
        new Benchmark().runBenchmark(args);
    }

    @Test
    public void testBenchmark() {
        String[] args = {
            "-a", "resnet", "-s", "1,3,224,224", "-c", "2", "-r", "{'layers':'18','flavor':'v1'}"
        };
        new Benchmark().runBenchmark(args);
    }

    @Test
    public void testMultithreadedBenchmark() {
        System.setProperty("collect-memory", "true");
        try {
            String[] args = {
                "-a",
                "resnet",
                "-s",
                "(1,3,224,224)f",
                "-d",
                "1",
                "-c",
                "2",
                "-r",
                "{'layers':'18','flavor':'v1'}",
                "-t",
                "-1"
            };
            Benchmark.main(args);
        } finally {
            System.clearProperty("collect-memory");
        }
    }
}
