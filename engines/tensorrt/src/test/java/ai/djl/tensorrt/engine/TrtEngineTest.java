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
package ai.djl.tensorrt.engine;

import ai.djl.engine.Engine;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class TrtEngineTest {

    @Test
    public void getVersion() {
        String version;
        try {
            Engine engine = Engine.getEngine("TensorRT");
            version = engine.getVersion();
        } catch (Exception ignore) {
            throw new SkipException("Your os configuration doesn't support TensorRT.");
        }
        Assert.assertEquals(version, "8.0.1");
    }
}
