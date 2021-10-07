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
import ai.djl.engine.StandardCapabilities;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.GradientCollector;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * The {@code OrtEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://microsoft.github.io/onnxruntime/">ONNX Runtime Deep Learning Library</a>.
 *
 * <p>To get an instance of the {@code OrtEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "OnnxRuntime".
 */
public final class OrtEngine extends Engine {

    public static final String ENGINE_NAME = "OnnxRuntime";
    static final int RANK = 10;

    private OrtEnvironment env;
    private Engine alternativeEngine;
    private boolean initialized;

    private OrtEngine() {
        // init OrtRuntime
        this.env = OrtEnvironment.getEnvironment();
    }

    static Engine newInstance() {
        return new OrtEngine();
    }

    /** {@inheritDoc} */
    @Override
    public Engine getAlternativeEngine() {
        if (!initialized && !Boolean.getBoolean("ai.djl.onnx.disable_alternative")) {
            Engine engine = Engine.getInstance();
            if (engine.getRank() < getRank()) {
                // alternativeEngine should not have the same rank as OnnxRuntime
                alternativeEngine = engine;
            }
            initialized = true;
        }
        return alternativeEngine;
    }

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return RANK;
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return "1.9.0";
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        if (StandardCapabilities.MKL.equals(capability)) {
            return true;
        } else if (StandardCapabilities.CUDA.equals(capability)) {
            try {
                OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
                sessionOptions.addCUDA();
                return true;
            } catch (OrtException e) {
                return false;
            }
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return new OrtModel(name, newBaseManager(device), env);
    }

    /** {@inheritDoc} */
    @Override
    public SymbolBlock newSymbolBlock(NDManager manager) {
        throw new UnsupportedOperationException("ONNXRuntime does not support empty SymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return newBaseManager(null);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return OrtNDManager.getSystemManager().newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append(getEngineName()).append(':').append(getVersion()).append(", ");
        sb.append(getEngineName())
                .append(':')
                .append(getVersion())
                .append(", capabilities: [\n\t" + StandardCapabilities.MKL + ",\n");
        if (hasCapability(StandardCapabilities.CUDA)) {
            sb.append("\t").append(StandardCapabilities.CUDA).append(",\n"); // NOPMD
        }
        return sb.toString();
    }
}
