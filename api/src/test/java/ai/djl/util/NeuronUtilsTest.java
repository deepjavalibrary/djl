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
package ai.djl.util;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class NeuronUtilsTest {

    @Test
    public void testNeuronUtils() throws IOException {
        Path dir = Paths.get("build/neuron_device/");
        try {
            for (int i = 0; i < 4; ++i) {
                Path nd = dir.resolve("neuron" + i);
                for (int j = 0; j < 2; ++j) {
                    Path nc = nd.resolve("neuron_core" + j);
                    Files.createDirectories(nc);
                }
            }
            NeuronUtils.hasNeuron();
            Assert.assertEquals(NeuronUtils.getNeuronCores(dir.toString()), 8);
        } finally {
            Utils.deleteQuietly(dir);
        }
    }
}
