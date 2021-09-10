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
package ai.djl.dlr.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.dlr.jni.JniUtils;
import ai.djl.dlr.jni.LibUtils;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.GradientCollector;

/**
 * The {@code DlrEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://github.com/neo-ai/neo-ai-dlr">Neo DLR</a>.
 *
 * <p>To get an instance of the {@code DlrEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "DLR".
 */
public final class DlrEngine extends Engine {

    public static final String ENGINE_NAME = "DLR";
    static final int RANK = 10;

    private Engine alternativeEngine;
    private boolean initialized;

    private DlrEngine() {}

    static Engine newInstance() {
        try {
            LibUtils.loadLibrary();
            return new DlrEngine();
        } catch (Throwable t) {
            throw new EngineException("Failed to load DLR native library", t);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Engine getAlternativeEngine() {
        if (!initialized && !Boolean.getBoolean("ai.djl.dlr.disable_alternative")) {
            Engine engine = Engine.getInstance();
            if (engine.getRank() < getRank()) {
                // alternativeEngine should not have the same rank as DLR
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
        return JniUtils.getDlrVersion();
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public SymbolBlock newSymbolBlock(NDManager manager) {
        throw new UnsupportedOperationException("DLR does not support empty SymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        // Only support CPU for now
        if (device != null && device != Device.cpu()) {
            throw new IllegalArgumentException("DLR only support CPU");
        }
        return new DlrModel(name, newBaseManager(Device.cpu()));
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return newBaseManager(null);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return DlrNDManager.getSystemManager().newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        throw new UnsupportedOperationException("Not supported for DLR");
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {
        throw new UnsupportedOperationException("Not supported for DLR");
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return getEngineName() + ':' + getVersion();
    }
}
