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
import ai.djl.TrainingDivergedException;
import ai.djl.engine.Engine;
import ai.djl.engine.StandardCapabilities;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.Assertions;
import org.testng.Assert;

public final class TestUtils {

    private TestUtils() {}

    public static boolean isWindows() {
        return System.getProperty("os.name").startsWith("Win");
    }

    public static boolean isMxnet() {
        Engine engine = Engine.getInstance();
        return "MXNet".equals(engine.getEngineName());
    }

    public static boolean isEngine(String name) {
        Engine engine = Engine.getInstance();
        return name.equals(engine.getEngineName());
    }

    public static void verifyNDArrayValues(
            NDArray array, Shape expectedShape, float sum, float mean, float max, float min) {
        if (array.isNaN().any().getBoolean()) {
            throw new TrainingDivergedException("There are NANs in this array");
        }
        Assert.assertEquals(array.getShape(), expectedShape);
        Assertions.assertAlmostEquals(array.sum().getFloat(), sum);
        Assertions.assertAlmostEquals(array.mean().getFloat(), mean);
        Assertions.assertAlmostEquals(array.max().getFloat(), max);
        Assertions.assertAlmostEquals(array.min().getFloat(), min);
    }

    public static Device[] getDevices() {
        if (!Engine.getInstance().hasCapability(StandardCapabilities.CUDNN)
                && TestUtils.isMxnet()) {
            return new Device[] {
                Device.cpu()
            }; // TODO: RNN is not implemented on MXNet without cuDNN
        }
        return Engine.getInstance().getDevices(1);
    }
}
