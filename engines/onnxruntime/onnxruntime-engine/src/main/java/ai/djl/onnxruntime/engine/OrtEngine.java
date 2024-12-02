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
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@code OrtEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://microsoft.github.io/onnxruntime/">ONNX Runtime Deep Learning Library</a>.
 *
 * <p>To get an instance of the {@code OrtEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "OnnxRuntime".
 */
public final class OrtEngine extends Engine {

    private static final Logger logger = LoggerFactory.getLogger(OrtEngine.class);

    public static final String ENGINE_NAME = "OnnxRuntime";
    static final int RANK = 10;

    private OrtEnvironment env;
    private Engine alternativeEngine;
    private boolean initialized;

    private OrtEngine() {
        // init OrtRuntime
        OrtEnvironment.ThreadingOptions options = new OrtEnvironment.ThreadingOptions();
        try {
            Integer interOpThreads = Integer.getInteger("ai.djl.onnxruntime.num_interop_threads");
            Integer intraOpsThreads = Integer.getInteger("ai.djl.onnxruntime.num_threads");
            if (interOpThreads != null) {
                options.setGlobalInterOpNumThreads(interOpThreads);
            }
            if (intraOpsThreads != null) {
                options.setGlobalIntraOpNumThreads(intraOpsThreads);
            }
            OrtLoggingLevel logging = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
            String name = OrtEnvironment.DEFAULT_NAME;
            this.env = OrtEnvironment.getEnvironment(logging, name, options);
        } catch (OrtException e) {
            options.close();
            throw new AssertionError("Failed to config OrtEnvironment", e);
        }
    }

    static Engine newInstance() {
        return new OrtEngine();
    }

    OrtEnvironment getEnv() {
        return env;
    }

    /** {@inheritDoc} */
    @Override
    public Engine getAlternativeEngine() {
        if (!initialized && !Boolean.getBoolean("ai.djl.onnx.disable_alternative")) {
            Engine engine;
            if (Engine.hasEngine("PyTorch")) {
                // workaround MXNet engine issue on CI
                engine = Engine.getEngine("PyTorch");
            } else {
                engine = Engine.getInstance();
            }
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
        return "1.20.0";
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        if (StandardCapabilities.MKL.equals(capability)) {
            return true;
        } else if (StandardCapabilities.CUDA.equals(capability)) {
            try (OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions()) {
                sessionOptions.addCUDA();
                return true;
            } catch (OrtException e) {
                logger.warn("CUDA is not supported OnnxRuntime engine: {}", e.getMessage());
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
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append(getEngineName()).append(':').append(getVersion()).append(", ");
        sb.append(getEngineName())
                .append(':')
                .append(getVersion())
                .append(", capabilities: [\n\t" + StandardCapabilities.MKL);
        if (hasCapability(StandardCapabilities.CUDA)) {
            sb.append(",\n\t").append(StandardCapabilities.CUDA); // NOPMD
        }
        sb.append(']');
        return sb.toString();
    }
}
