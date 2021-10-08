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
package ai.djl;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The {@code Device} class provides the specified assignment for CPU/GPU processing on the {@code
 * NDArray}.
 *
 * <p>Users can use this to specify whether to load/compute the {@code NDArray} on CPU/GPU with
 * deviceType and deviceId provided.
 *
 * @see <a href="https://d2l.djl.ai/chapter_deep-learning-computation/use-gpu.html">The D2L chapter
 *     on GPU devices</a>
 */
public final class Device {

    private static final Map<String, Device> CACHE = new ConcurrentHashMap<>();

    private static final Device CPU = new Device(Type.CPU, -1);
    private static final Device GPU = Device.of(Type.GPU, 0);

    private String deviceType;
    private int deviceId;

    /**
     * Creates a {@code Device} with basic information.
     *
     * @param deviceType the device type, typically CPU or GPU
     * @param deviceId the deviceId on the hardware. For example, if you have multiple GPUs, you can
     *     choose which GPU to process the NDArray
     */
    private Device(String deviceType, int deviceId) {
        this.deviceType = deviceType;
        this.deviceId = deviceId;
    }

    /**
     * Returns a {@code Device} with device type and device id.
     *
     * @param deviceType the device type, typically CPU or GPU
     * @param deviceId the deviceId on the hardware.
     * @return a {@code Device} instance
     */
    public static Device of(String deviceType, int deviceId) {
        if (Type.CPU.equals(deviceType)) {
            return CPU;
        }
        String key = deviceType + '-' + deviceId;
        return CACHE.computeIfAbsent(key, k -> new Device(deviceType, deviceId));
    }

    /**
     * Returns the device type of the Device.
     *
     * @return the device type of the Device
     */
    public String getDeviceType() {
        return deviceType;
    }

    /**
     * Returns the {@code deviceId} of the Device.
     *
     * @return the {@code deviceId} of the Device
     */
    public int getDeviceId() {
        return deviceId;
    }

    /**
     * Returns if the {@code Device} is GPU.
     *
     * @return if the {@code Device} is GPU.
     */
    public boolean isGpu() {
        return Type.GPU.equals(deviceType);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (Type.CPU.equals(deviceType)) {
            return deviceType + "()";
        }
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
        Device device = (Device) o;
        if (Type.CPU.equals(deviceType)) {
            return Objects.equals(deviceType, device.deviceType);
        }
        return deviceId == device.deviceId && Objects.equals(deviceType, device.deviceType);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(deviceType, deviceId);
    }

    /**
     * Returns the default CPU Device.
     *
     * @return the default CPU Device
     */
    public static Device cpu() {
        return CPU;
    }

    /**
     * Returns the default GPU Device.
     *
     * @return the default GPU Device
     */
    public static Device gpu() {
        return GPU;
    }

    /**
     * Returns a new instance of GPU {@code Device} with the specified {@code deviceId}.
     *
     * @param deviceId the GPU device ID
     * @return a new instance of GPU {@code Device} with specified {@code deviceId}
     */
    public static Device gpu(int deviceId) {
        return of(Type.GPU, deviceId);
    }

    /** Contains device type string constants. */
    public interface Type {
        String CPU = "cpu";
        String GPU = "gpu";
    }
}
