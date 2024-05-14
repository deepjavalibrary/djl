/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.integration;

import ai.djl.Device;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.TestRequirements;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;

public class MpsTest {

    @Test(enabled = false)
    public void testMps() {
        TestRequirements.macosM1();

        Device device = Device.of("mps", -1);
        try (NDManager manager = NDManager.newBaseManager(device)) {
            NDArray array = manager.zeros(new Shape(1, 2));
            Assert.assertEquals(array.getDevice().getDeviceType(), "mps");
        }
    }

    @Test(enabled = false)
    public void testToTensorMPS() {
        TestRequirements.macosM1();

        // Test that toTensor does not fail on MPS (e.g. due to use of float64 for division)
        try (NDManager manager = NDManager.newBaseManager(Device.fromName("mps"))) {
            NDArray array = manager.create(127f).reshape(1, 1, 1, 1);
            NDArray tensor = array.getNDArrayInternal().toTensor();
            Assert.assertEquals(tensor.toFloatArray(), new float[] {127f / 255f});
        }
    }

    @Test(enabled = false)
    public void testClassificationsMPS() {
        TestRequirements.macosM1();

        // Test that classifications do not fail on MPS (e.g. due to conversion of probabilities to
        // float64)
        try (NDManager manager = NDManager.newBaseManager(Device.fromName("mps"))) {
            List<String> names = Arrays.asList("First", "Second", "Third", "Fourth", "Fifth");
            NDArray tensor = manager.create(new float[] {0f, 0.125f, 1f, 0.5f, 0.25f});
            Classifications classifications = new Classifications(names, tensor);
            Assert.assertEquals(classifications.best().getClassName(), "Third");
        }
    }
}
