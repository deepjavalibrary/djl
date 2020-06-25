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
package ai.djl.test.mock;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;

public class MockEngine extends Engine {

    private String version;

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return new MockModel();
    }

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return "MockEngine";
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return version;
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return new MockNDManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return new MockNDManager();
    }

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {}

    public void setVersion(String version) {
        this.version = version;
    }
}
