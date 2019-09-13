/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package org.apache.mxnet.engine;

import java.lang.management.MemoryUsage;
import org.apache.mxnet.jna.JnaUtils;
import software.amazon.ai.Context;
import software.amazon.ai.Model;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.GradientCollector;
import software.amazon.ai.training.ParameterStore;
import software.amazon.ai.training.optimizer.Optimizer;

public class MxEngine extends Engine {

    public static final String ENGINE_NAME = "MXNet";

    MxEngine() {
        // Workaround MXNet engine lazy initialization issue
        JnaUtils.getAllOpNames();

        JnaUtils.setNumpyMode(true);
    }

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public int getGpuCount() {
        return JnaUtils.getGpuCount();
    }

    /** {@inheritDoc} */
    @Override
    public MemoryUsage getGpuMemory(Context context) {
        long[] mem = JnaUtils.getGpuMemory(context);
        long committed = mem[1] - mem[0];
        return new MemoryUsage(-1, committed, committed, mem[1]);
    }

    /** {@inheritDoc} */
    @Override
    public Context defaultContext() {
        if (getGpuCount() > 0) {
            return Context.gpu();
        }
        return Context.cpu();
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        int version = JnaUtils.getVersion();
        int major = version / 10000;
        int minor = version / 100 - major * 100;
        int patch = version % 100;

        return major + "." + minor + '.' + patch;
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(Context context) {
        return new MxModel(context);
    }

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        return new MxGradientCollector();
    }

    /** {@inheritDoc} */
    @Override
    public ParameterStore newParameterStore(Optimizer optimizer, boolean aggregateOnGPU) {
        return new MxParameterStore(aggregateOnGPU, optimizer);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return MxNDManager.getSystemManager().newSubManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Context context) {
        return MxNDManager.getSystemManager().newSubManager();
    }

    /**
     * Sets whether to run the MxEngine in numpy mode.
     *
     * @param numpy True to use numpy mode
     */
    public void setNumpyMode(boolean numpy) {
        // Helper to avoid race condition with MxEngine initialization
        JnaUtils.setNumpyMode(numpy);
    }
}
