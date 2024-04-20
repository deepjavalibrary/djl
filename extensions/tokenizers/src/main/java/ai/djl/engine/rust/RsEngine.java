/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.engine.rust;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.huggingface.tokenizers.jni.LibUtils;
import ai.djl.ndarray.NDManager;

/** The {@code RsEngine} is an implementation of the {@link Engine} rust engine. */
public final class RsEngine extends Engine {

    public static final String ENGINE_NAME = "Rust";
    static final int RANK = 3;

    private RsEngine() {}

    @SuppressWarnings("PMD.AvoidRethrowingException")
    static Engine newInstance() {
        try {
            LibUtils.checkStatus();
            return new RsEngine();
        } catch (EngineException e) {
            throw e;
        } catch (Throwable t) {
            throw new EngineException("Failed to load Rust native library", t);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Engine getAlternativeEngine() {
        return null;
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
        return Engine.getDjlVersion();
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return new RsModel(name, device);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return RsNDManager.getSystemManager().newSubManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return RsNDManager.getSystemManager().newSubManager(device);
    }
}
