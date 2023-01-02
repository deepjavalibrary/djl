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
import ai.djl.engine.StandardCapabilities;

import org.testng.SkipException;

import java.util.Arrays;

public final class TestUtils {

    private static String engineName = Engine.getDefaultEngineName();

    private TestUtils() {}

    public static void setEngine(String engineName) {
        TestUtils.engineName = engineName;
    }

    public static String getEngine() {
        return engineName;
    }

    public static boolean isWindows() {
        return System.getProperty("os.name").startsWith("Win");
    }

    public static void requiresEngine(String... engines) {
        for (String e : engines) {
            if (engineName.equals(e)) {
                return;
            }
        }
        throw new SkipException(
                "This test requires one of the engines: " + Arrays.toString(engines));
    }

    public static Device[] getDevices(int maxGpus) {
        Engine engine = Engine.getEngine(engineName);
        if (!engine.hasCapability(StandardCapabilities.CUDNN) && "MXNet".equals(engineName)) {
            return new Device[] {
                Device.cpu()
            }; // TODO: RNN is not implemented on MXNet without cuDNN
        }
        return engine.getDevices(maxGpus);
    }
}
