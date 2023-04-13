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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class NeuronUtilsTest {

    private static final Logger logger = LoggerFactory.getLogger(NeuronUtilsTest.class);

    @Test
    public void testNeuronUtils() throws IOException {
        Path dev = Paths.get("build/dev/");
        Path sys = Paths.get("build/neuron_device/");
        try {
            Files.createDirectories(dev);
            for (int i = 1; i < 4; ++i) {
                Path nd = dev.resolve("neuron" + i);
                Files.createFile(nd);

                Path vd = sys.resolve("neuron" + i);
                Files.createDirectories(vd);
                for (int j = 0; j < 2; ++j) {
                    Path nc = vd.resolve("neuron_core" + j);
                    Files.createDirectories(nc);
                }
            }

            logger.info("hasNeuron: {}", NeuronUtils.hasNeuron());
            logger.info("# neuron cores: {}", NeuronUtils.getNeuronCores());
            List<Path> devices = NeuronUtils.getNeuronDevices(dev.toString());
            Assert.assertEquals(devices.size(), 3);
            String vd = sys.resolve("neuron1").toString();
            Assert.assertEquals(NeuronUtils.getNeuronCoresPerDevice(vd), 2);
        } finally {
            Utils.deleteQuietly(sys);
            Utils.deleteQuietly(dev);
        }
    }
}
