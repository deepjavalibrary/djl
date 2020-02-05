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
package ai.djl.pytorch.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.jni.JniUtils;

/**
 * The {@code PtEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://pytorch.org/">PyTorch Deep Learning Framework</a>.
 *
 * <p>To get an instance of the {@code PtEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "PyTorch".
 */
public class PtEngine extends Engine {

    public static final String ENGINE_NAME = "PyTorch";

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return JniUtils.libraryVersion();
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(Device device) {
        return new PtModel(device);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return PtNDManager.getSystemManager().newSubManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return PtNDManager.getSystemManager().newSubManager();
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {}
}
