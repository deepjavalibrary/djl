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

package ai.djl.onnxruntime.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

/**
 * The {@code OrtEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://microsoft.github.io/onnxruntime/">ONNX Runtime Deep Learning Library</a>.
 *
 * <p>To get an instance of the {@code OrtEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "OnnxRuntime".
 */
public final class OrtEngine extends Engine implements AutoCloseable {

    public static final String ENGINE_NAME = "OnnxRuntime";
    private OrtEnvironment environment;

    private OrtEngine(OrtEnvironment env) {
        environment = env;
    }

    static Engine newInstance() {
        LibUtils.prepareLibrary();
        // init OrtRuntime
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        return new OrtEngine(environment);
    }

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return "1.3.0";
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return null;
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
    public GradientCollector newGradientCollector() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {}

    /** {@inheritDoc} */
    @Override
    public void close() throws OrtException {
        environment.close();
    }
}
