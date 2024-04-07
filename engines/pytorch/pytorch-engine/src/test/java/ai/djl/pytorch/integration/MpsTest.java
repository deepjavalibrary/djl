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

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;

public class MpsTest {

    @Test
    public void testMps() {
        if (!"aarch64".equals(System.getProperty("os.arch"))
                || !System.getProperty("os.name").startsWith("Mac")) {
            throw new SkipException("MPS test requires M1 macOS.");
        }

        Device device = Device.of("mps", -1);
        try (NDManager manager = NDManager.newBaseManager(device)) {
            NDArray array = manager.zeros(new Shape(1, 2));
            Assert.assertEquals(array.getDevice().getDeviceType(), "mps");
        }
    }

    private static boolean checkMpsCompatible() {
        return "aarch64".equals(System.getProperty("os.arch"))
                && System.getProperty("os.name").startsWith("Mac");
    }

    @Test
    public void testToTensorMPS() {
        if (!checkMpsCompatible()) {
            throw new SkipException("MPS toTensor test requires Apple Silicon macOS.");
        }

        // Test that toTensor does not fail on MPS (e.g. due to use of float64 for division)
        try (NDManager manager = NDManager.newBaseManager(Device.fromName("mps"))) {
            NDArray array = manager.create(127f).reshape(1, 1, 1, 1);
            NDArray tensor = array.getNDArrayInternal().toTensor();
            Assert.assertEquals(tensor.toFloatArray(), new float[] {127f / 255f});
        }
    }

    @Test
    public void testClassificationsMPS() {
        if (!checkMpsCompatible()) {
            throw new SkipException("MPS classification test requires Apple Silicon macOS.");
        }

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
