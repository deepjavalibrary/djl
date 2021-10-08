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

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.engine.StandardCapabilities;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.SymbolBlock;
import ai.djl.tensorrt.jni.JniUtils;
import ai.djl.tensorrt.jni.LibUtils;
import ai.djl.training.GradientCollector;
import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * The {@code TrtEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://github.com/NVIDIA/TensorRT">TensorRT</a>.
 *
 * <p>To get an instance of the {@code TrtEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "TensorRT".
 */
public final class TrtEngine extends Engine {

    public static final String ENGINE_NAME = "TensorRT";
    static final int RANK = 10;

    private Engine alternativeEngine;
    private boolean initialized;

    private TrtEngine() {}

    static Engine newInstance() {
        try {
            LibUtils.loadLibrary();
            JniUtils.initPlugins("");
            String paths = System.getenv("TENSORRT_EXTRA_LIBRARY_PATH");
            if (paths == null) {
                paths = System.getProperty("TENSORRT_EXTRA_LIBRARY_PATH");
            }
            if (paths != null) {
                String[] files = paths.split(",");
                for (String file : files) {
                    Path path = Paths.get(file);
                    if (Files.notExists(path)) {
                        throw new FileNotFoundException(
                                "TensorRT extra Library not found: " + file);
                    }
                    System.load(path.toAbsolutePath().toString()); // NOPMD
                }
            }
            return new TrtEngine();
        } catch (Throwable t) {
            throw new EngineException("Failed to load TensorRT native library", t);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Engine getAlternativeEngine() {
        if (!initialized && !Boolean.getBoolean("ai.djl.tensorrt.disable_alternative")) {
            Engine engine = Engine.getInstance();
            if (engine.getRank() < getRank()) {
                // alternativeEngine should not have the same rank as TensorRT
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
        return JniUtils.getTrtVersion();
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return StandardCapabilities.CUDA.equals(capability);
    }

    /** {@inheritDoc} */
    @Override
    public SymbolBlock newSymbolBlock(NDManager manager) {
        throw new UnsupportedOperationException("TensorRT does not support empty SymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return new TrtModel(name, newBaseManager(device));
    }

    /** {@inheritDoc} */
    @Override
    public TrtNDManager newBaseManager() {
        return newBaseManager(null);
    }

    /** {@inheritDoc} */
    @Override
    public TrtNDManager newBaseManager(Device device) {
        // Only support GPU for now
        device = device == null ? defaultDevice() : device;
        if (!device.isGpu()) {
            throw new IllegalArgumentException("TensorRT only support GPU");
        }
        return TrtNDManager.getSystemManager().newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        throw new UnsupportedOperationException("Not supported for TensorRT");
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {
        throw new UnsupportedOperationException("Not supported for TensorRT");
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append(getEngineName()).append(':').append(getVersion()).append(", ");
        if (alternativeEngine != null) {
            sb.append("Alternative engine: ").append(alternativeEngine.getEngineName());
        } else {
            sb.append("No alternative engine found");
        }
        return sb.toString();
    }
}
