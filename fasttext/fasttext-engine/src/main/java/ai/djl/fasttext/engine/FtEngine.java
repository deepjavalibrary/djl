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
package ai.djl.fasttext.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;

/**
 * The {@code FtEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://fasttext.cc//">Facebook fastText Framework</a>.
 *
 * <p>To get an instance of the {@code FtEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "fastText".
 */
public class FtEngine extends Engine {

    public static final String ENGINE_NAME = "fastText";

    /** Constructs an fastText Engine. */
    FtEngine() {}

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return "0.4";
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(Device device) {
        return new FtModel();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {}

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return "Name: " + getEngineName() + ", version: " + getVersion();
    }
}
