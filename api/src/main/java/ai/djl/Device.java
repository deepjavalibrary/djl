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

import ai.djl.engine.Engine;
import java.util.Objects;

/**
 * The {@code Device} class provides the specified assignment for CPU/GPU processing on the {@code
 * NDArray}.
 *
 * <p>Users can use this to specify whether to load/compute the {@code NDArray} on CPU/GPU with
 * deviceType and deviceId provided.
 */
public class Device {

    private static final Device CPU = new Device("cpu", 0);
    private static final Device GPU = new Device("gpu", 0);

    private String deviceType;
    private int deviceId;

    /**
     * Creates a {@code Device} with basic information.
     *
     * @param deviceType the device type, typically CPU or GPU
     * @param deviceId the deviceId on the hardware. For example, if you have multiple GPUs, you can
     *     choose which GPU to process the NDArray
     */
    public Device(String deviceType, int deviceId) {
        this.deviceType = deviceType;
        this.deviceId = deviceId;
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
        Device device = (Device) o;
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
     * Returns a new instance of CPU {@code Device} with the specified {@code deviceId}.
     *
     * @param deviceId the CPU device ID
     * @return a new instance of CPU {@code Device} with the specified {@code deviceId}
     */
    public static Device cpu(int deviceId) {
        return new Device("cpu", deviceId);
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
        return new Device("gpu", deviceId);
    }

    /**
     * Returns an array of devices given the maximum number of GPUs to use.
     *
     * <p>If GPUs are available, it will return an array of {@code Device} of size
     * \(min(numAvailable, maxGpus)\). Else, it will return an array with a single CPU device.
     *
     * @param maxGpus the max number of GPUs to use
     * @return an array of devices
     */
    public static Device[] getDevices(int maxGpus) {
        int count = Engine.getInstance().getGpuCount();
        if (maxGpus <= 0 || count <= 0) {
            return new Device[] {CPU};
        }
        count = Math.min(maxGpus, count);

        Device[] devices = new Device[count];
        for (int i = 0; i < count; ++i) {
            devices[i] = new Device("gpu", i);
        }
        return devices;
    }

    /**
     * Returns the default context used in Engine.
     *
     * <p>The default type is defined by whether the Deep Learning framework is recognizing GPUs
     * available on your machine. If there is no GPU available, CPU will be used.
     *
     * @return a {@link Device}
     */
    public static Device defaultDevice() {
        return Engine.getInstance().defaultDevice();
    }

    /**
     * Returns the given device or the default if it is null.
     *
     * @param device the device to try to return
     * @return the given device or the default if it is null
     */
    public static Device defaultIfNull(Device device) {
        if (device != null) {
            return device;
        }
        return Engine.getInstance().defaultDevice();
    }

    /**
     * Returns the given device or the passed in default if it is null.
     *
     * @param device the device to try to return
     * @param def the default device to return if device is null
     * @return the given device or the passed in default if it is null
     */
    public static Device defaultIfNull(Device device, Device def) {
        if (device != null) {
            return device;
        }
        return defaultIfNull(def);
    }
}
