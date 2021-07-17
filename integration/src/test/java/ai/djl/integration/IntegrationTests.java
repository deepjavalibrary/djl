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
package ai.djl.integration;

import ai.djl.util.cuda.CudaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

public class IntegrationTests {

    private static final Logger logger = LoggerFactory.getLogger(IntegrationTests.class);

    @Test
    public void runIntegrationTests() {
        String[] args = {};

        String[] engines;
        String defaultEngine = System.getProperty("ai.djl.default_engine");
        if (defaultEngine == null) {
            // TODO: windows CPU build is having OOM issue if 3 engines are loaded and running tests
            // together
            if (System.getProperty("os.name").startsWith("Win")) {
                engines = new String[] {"MXNet"};
            } else {
                engines = new String[] {"MXNet", "PyTorch", "TensorFlow", "XGBoost"};
            }
        } else {
            engines = new String[] {defaultEngine};
        }

        for (String engine : engines) {
            System.setProperty("ai.djl.default_engine", engine);
            logger.info("Testing engine: {} ...", engine);
            Assert.assertTrue(new IntegrationTest(IntegrationTest.class).runTests(args));
            // currently each engine will reserve a certain amount of memory and hold it until
            // process terminate so running 3 different engines sequentially without
            // calling System.exit() causes OOM issue. For GPU env, only defaultEngine is run
            if (CudaUtils.hasCuda()) {
                break;
            }
        }
    }
}
