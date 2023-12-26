/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.llama.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.llama.jni.LibUtils;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Platform;
import ai.djl.util.passthrough.PassthroughNDManager;

/** The {@code LlamaEngine} is an implementation of the {@link Engine} based on the llama.cpp. */
public final class LlamaEngine extends Engine {

    public static final String ENGINE_NAME = "Llama";
    static final int RANK = 10;

    private Engine alternativeEngine;
    private boolean initialized;

    private LlamaEngine() {
        try {
            LibUtils.loadLibrary();
        } catch (EngineException e) { // NOPMD
            throw e;
        } catch (Throwable t) {
            throw new EngineException("Failed to load llama.cpp native library", t);
        }
    }

    static Engine newInstance() {
        return new LlamaEngine();
    }

    /** {@inheritDoc} */
    @Override
    public Engine getAlternativeEngine() {
        if (!initialized && !Boolean.getBoolean("ai.djl.llama.disable_alternative")) {
            Engine engine = Engine.getInstance();
            if (engine.getRank() < getRank()) {
                // alternativeEngine should not have the same rank as Llama
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
        Platform platform = Platform.detectPlatform("llama");
        return platform.getVersion();
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return new LlamaModel(name, newBaseManager(device));
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return newBaseManager(null);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return PassthroughNDManager.INSTANCE;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return getEngineName() + ':' + getVersion() + ", " + getEngineName() + ':' + getVersion();
    }
}
