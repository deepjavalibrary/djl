/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ml.lightgbm;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.ml.lightgbm.jni.LibUtils;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.GradientCollector;

import java.io.IOException;

/**
 * The {@code LgbmEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://github.com/microsoft/LightGBM">LightGBM</a>.
 *
 * <p>To get an instance of the {@code LgbmEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "LightGBM".
 */
public final class LgbmEngine extends Engine {

    public static final String ENGINE_NAME = "LightGBM";
    public static final String ENGINE_VERSION = "3.2.110";
    static final int RANK = 10;

    private Engine alternativeEngine;
    private boolean initialized;

    private LgbmEngine() {
        try {
            LibUtils.loadNative();
        } catch (IOException e) {
            throw new EngineException("Failed to initialize LightGBMEngine", e);
        }
    }

    static Engine newInstance() {
        return new LgbmEngine();
    }

    /** {@inheritDoc} */
    @Override
    public Engine getAlternativeEngine() {
        if (!initialized && !Boolean.getBoolean("ai.djl.lightgbm.disable_alternative")) {
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
        return ENGINE_VERSION;
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public SymbolBlock newSymbolBlock(NDManager manager) {
        throw new UnsupportedOperationException("LightGBM does not support empty symbol block");
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return new LgbmModel(name, newBaseManager(device));
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return newBaseManager(null);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return LgbmNDManager.getSystemManager().newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        throw new UnsupportedOperationException("Not supported for LightGBM");
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {
        throw new UnsupportedOperationException("Not supported for LightGBM");
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return getEngineName() + ':' + getVersion();
    }
}
