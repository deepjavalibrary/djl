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
package ai.djl.examples.inference;

import ai.djl.examples.inference.benchmark.Benchmark;
import ai.djl.examples.inference.benchmark.MultithreadedBenchmark;
import org.testng.annotations.Test;

public class BenchmarkTest {

    @Test
    public void testBenchmark() {
        String[] args = {
            "-c",
            "2",
            "-i",
            "/Volumes/Unix/projects/Joule/examples/src/test/resources/segmentation.jpg",
            "-r",
            "{'layers':'18','flavor':'v1'}"
        };
        new Benchmark().runBenchmark(args);
    }

    @Test
    public void testMultithreadedBenchmark() {
        String[] args = {
            "-c",
            "2",
            "-i",
            "/Volumes/Unix/projects/Joule/examples/src/test/resources/segmentation.jpg",
            "-r",
            "{'layers':'18','flavor':'v1'}",
            "-t",
            "2"
        };
        new MultithreadedBenchmark().runBenchmark(args);
    }
}
