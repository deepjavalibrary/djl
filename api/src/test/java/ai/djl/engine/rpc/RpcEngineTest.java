/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.engine.rpc;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.util.passthrough.PassthroughNDManager;

import org.testng.Assert;
import org.testng.annotations.Test;

public class RpcEngineTest {

    @Test
    public void testRpcEngine() {
        Engine engine = RpcEngine.getEngine(RpcEngine.ENGINE_NAME);
        Assert.assertEquals(engine.getEngineName(), RpcEngine.ENGINE_NAME);
        Assert.assertEquals(engine.getRank(), 15);
        Assert.assertEquals(
                engine.getVersion(), Engine.class.getPackage().getSpecificationVersion());
        Assert.assertFalse(engine.hasCapability("CUDA"));
        try (NDManager manager = engine.newBaseManager()) {
            Assert.assertTrue(manager instanceof PassthroughNDManager);
        }
        Engine alt = engine.getAlternativeEngine();
        Assert.assertNotNull(alt);
    }
}
