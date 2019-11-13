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
package ai.djl.mxnet.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.ndarray.NDManager;
import java.lang.management.MemoryUsage;

/**
 * The {@code MxEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://mxnet.apache.org/">Apache MXNet Deep Learning Framework</a>.
 *
 * <p>To get an instance of the {@code MxEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "MXNet".
 */
public class MxEngine extends Engine {

    public static final String ENGINE_NAME = "MXNet";

    /** Constructs an MXNet Engine. */
    MxEngine() {
        // Workaround MXNet engine lazy initialization issue
        JnaUtils.getAllOpNames();

        JnaUtils.setNumpyMode(JnaUtils.NumpyMode.GLOBAL_ON);

        // Workaround MXNet shutdown crash issue
        Runtime.getRuntime().addShutdownHook(new Thread(JnaUtils::waitAll)); // NOPMD
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
    public MemoryUsage getGpuMemory(Device device) {
        long[] mem = JnaUtils.getGpuMemory(device);
        long committed = mem[1] - mem[0];
        return new MemoryUsage(-1, committed, committed, mem[1]);
    }

    /** {@inheritDoc} */
    @Override
    public Device defaultDevice() {
        if (getGpuCount() > 0) {
            return Device.gpu();
        }
        return Device.cpu();
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
    public Model newModel(Device device) {
        return new MxModel(device);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return MxNDManager.getSystemManager().newSubManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return MxNDManager.getSystemManager().newSubManager();
    }
}
