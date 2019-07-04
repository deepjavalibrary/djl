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
package software.amazon.ai;

import java.util.Objects;
import software.amazon.ai.engine.Engine;

/**
 * The {@code Context} class provides the specified assignment for CPU/GPU processing on the
 * NDArray.
 *
 * <p>Users can use this to specify whether to load/compute the NDArray on CPU/GPU with deviceType
 * and deviceId provided
 */
public class Context {

    private static final Context CPU = new Context("cpu", 0);
    private static final Context GPU = new Context("gpu", 0);

    private String deviceType;
    private int deviceId;

    /**
     * Creates {@code Context} with basic information.
     *
     * @param deviceType device type user would like to use, typically CPU or GPU
     * @param deviceId deviceId on the hardware. For example, if you have multiple GPUs, you can
     *     choose which GPU to process the NDArray
     */
    public Context(String deviceType, int deviceId) {
        this.deviceType = deviceType;
        this.deviceId = deviceId;
    }

    /**
     * Returns device type of the Context.
     *
     * @return device type of the Context
     */
    public String getDeviceType() {
        return deviceType;
    }

    /**
     * Returns {@code deviceId} of the Context.
     *
     * @return {@code deviceId} of the Context
     */
    public int getDeviceId() {
        return deviceId;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return deviceType + '(' + deviceId + ')';
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Context context = (Context) o;
        return deviceId == context.deviceId && Objects.equals(deviceType, context.deviceType);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(deviceType, deviceId);
    }

    /**
     * Returns default CPU Context.
     *
     * @return default CPU Context
     */
    public static Context cpu() {
        return CPU;
    }

    /**
     * Returns a new instance of CPU Context with specified {@code deviceId}.
     *
     * @param deviceId CPU device ID
     * @return a new instance of CPU Context with specified {@code deviceId}
     */
    public static Context cpu(int deviceId) {
        return new Context("cpu", deviceId);
    }

    /**
     * Returns default GPU Context.
     *
     * @return default GPU Context
     */
    public static Context gpu() {
        return GPU;
    }

    /**
     * Returns a new instance of GPU Context with specified {@code deviceId}.
     *
     * @param deviceId GPU device ID
     * @return a new instance of GPU Context with specified {@code deviceId}
     */
    public static Context gpu(int deviceId) {
        return new Context("gpu", deviceId);
    }

    /**
     * Returns the default context used in Engine
     *
     * <p>default type is defined by whether the Deep Learning framework is recognizing GPUs
     * available on your machine. If there is no GPU available, CPU will be used.
     *
     * @return {@link Context}
     */
    public static Context defaultContext() {
        return Engine.getInstance().defaultContext();
    }
}
