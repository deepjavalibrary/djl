/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import org.testng.Assert;
import org.testng.annotations.Test;

public class NDListGeneratorTest {

    @Test
    public void testHelp() {
        String[] args = {"ndlist-gen", "-h"};
        Benchmark.main(args);
    }

    @Test
    public void testMissingOptions() {
        String[] args = {"ndlist-gen", "-s"};
        boolean success = NDListGenerator.generate(args);
        Assert.assertFalse(success);
    }

    @Test
    public void testOnes() {
        String[] args = {"ndlist-gen", "-s", "1", "-o", "build/ones.ndlist", "-1"};
        boolean success = NDListGenerator.generate(args);
        Assert.assertTrue(success);
    }

    @Test
    public void testZeros() {
        String[] args = {"ndlist-gen", "-s", "1", "-o", "build/ones.ndlist"};
        boolean success = NDListGenerator.generate(args);
        Assert.assertTrue(success);
    }
}
