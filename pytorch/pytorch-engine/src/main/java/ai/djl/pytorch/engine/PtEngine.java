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
package ai.djl.pytorch.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.pytorch.jni.LibUtils;
import ai.djl.training.GradientCollector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@code PtEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://pytorch.org/">PyTorch Deep Learning Framework</a>.
 *
 * <p>To get an instance of the {@code PtEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "PyTorch".
 */
public final class PtEngine extends Engine {

    private static final Logger logger = LoggerFactory.getLogger(PtEngine.class);

    public static final String ENGINE_NAME = "PyTorch";

    private PtEngine() {}

    static Engine newInstance() {
        try {
            LibUtils.loadLibrary();
            if (Integer.getInteger("ai.djl.pytorch.num_interop_threads") != null) {
                JniUtils.setNumInteropThreads(
                        Integer.getInteger("ai.djl.pytorch.num_interop_threads"));
            }
            if (Integer.getInteger("ai.djl.pytorch.num_threads") != null) {
                JniUtils.setNumThreads(Integer.getInteger("ai.djl.pytorch.num_threads"));
            }
            return new PtEngine();
        } catch (Throwable t) {
            logger.warn("Failed to load PyTorch native library", t);
        }
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return "1.5.0";
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return JniUtils.getFeatures().contains(capability);
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return new PtModel(name, device);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return PtNDManager.getSystemManager().newSubManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return PtNDManager.getSystemManager().newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        return new PtGradientCollector();
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {
        JniUtils.setSeed(seed);
    }
}
