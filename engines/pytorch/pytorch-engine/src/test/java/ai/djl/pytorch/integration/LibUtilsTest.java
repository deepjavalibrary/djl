/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.engine.Engine;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class LibUtilsTest {

    @BeforeClass
    public void setup() {
        System.setProperty(
                "ai.djl.pytorch.native_helper", "ai.djl.pytorch.integration.LibUtilsTest");
    }

    @AfterClass
    public void teardown() {
        System.clearProperty("ai.djl.pytorch.native_helper");
    }

    @Test
    public void test() {
        Engine.getInstance();
    }

    public static void load(String path) {
        System.load(path); // NOPMD
    }
}
