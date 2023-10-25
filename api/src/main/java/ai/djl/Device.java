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

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
public class Device {

    private static final Map<String, Device> CACHE = new ConcurrentHashMap<>();

    private static final Device CPU = new Device(Type.CPU, -1);
    private static final Device GPU = Device.of(Type.GPU, 0);

    private static final Pattern DEVICE_NAME = Pattern.compile("([a-z]+)([0-9]*)");

    protected String deviceType;
    protected int deviceId;

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
     * Parses a deviceName string into a device for the default engine.
     *
     * @param deviceName deviceName String to parse
     * @return the parsed device
     * @see #fromName(String, Engine)
     */
    public static Device fromName(String deviceName) {
        return fromName(deviceName, Engine.getInstance());
    }

    /**
     * Parses a deviceName string into a device.
     *
     * <p>The main format of a device name string is "cpu", "gpu0", or "nc1". This is simply
     * deviceType concatenated with the deviceId. If no deviceId is used, -1 will be assumed.
     *
     * <p>There are also several simplified formats. The "-1", deviceNames corresponds to cpu.
     * Non-negative integer deviceNames such as "0", "1", or "2" correspond to gpus with those
     * deviceIds.
     *
     * <p>Finally, unspecified deviceNames (null or "") are parsed into the engine's default device.
     *
     * @param deviceName deviceName string
     * @param engine the engine the devie is for
     * @return the device
     */
    public static Device fromName(String deviceName, Engine engine) {
        if (deviceName == null || deviceName.isEmpty()) {
            return engine.defaultDevice();
        }

        if (deviceName.contains("+")) {
            String[] split = deviceName.split("\\+");
            List<Device> subDevices =
                    Arrays.stream(split).map(n -> fromName(n, engine)).collect(Collectors.toList());
            return new MultiDevice(subDevices);
        }

        Matcher matcher = DEVICE_NAME.matcher(deviceName);
        if (matcher.matches()) {
            String deviceType = matcher.group(1);
            int deviceId = -1;
            if (!matcher.group(2).isEmpty()) {
                deviceId = Integer.parseInt(matcher.group(2));
            }
            return Device.of(deviceType, deviceId);
        }

        try {
            int deviceId = Integer.parseInt(deviceName);

            if (deviceId < 0) {
                return Device.cpu();
            }
            return Device.gpu(deviceId);
        } catch (NumberFormatException ignored) {
        }
        throw new IllegalArgumentException("Failed to parse device name: " + deviceName);
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

    /** A combined {@link Device} representing the composition of multiple other devices. */
    public static class MultiDevice extends Device {

        List<Device> devices;

        /**
         * Constructs a {@link MultiDevice} with a range of new devices.
         *
         * @param deviceType the type of the sub-devices
         * @param startInclusive the start (inclusive) of the devices range
         * @param endExclusive the end (exclusive) of the devices range
         */
        public MultiDevice(String deviceType, int startInclusive, int endExclusive) {
            this(
                    IntStream.range(startInclusive, endExclusive)
                            .mapToObj(i -> Device.of(deviceType, i))
                            .collect(Collectors.toList()));
        }

        /**
         * Constructs a {@link MultiDevice} from sub devices.
         *
         * @param devices the sub devices
         */
        public MultiDevice(Device... devices) {
            this(Arrays.asList(devices));
        }

        /**
         * Constructs a {@link MultiDevice} from sub devices.
         *
         * @param devices the sub devices
         */
        public MultiDevice(List<Device> devices) {
            super(null, -1);
            devices.sort(
                    Comparator.comparing(Device::getDeviceType, String.CASE_INSENSITIVE_ORDER)
                            .thenComparingInt(Device::getDeviceId));
            this.deviceType =
                    String.join(
                            "+",
                            (Iterable<String>)
                                    () ->
                                            devices.stream()
                                                    .map(d -> d.getDeviceType() + d.getDeviceId())
                                                    .iterator());
            this.devices = devices;
        }

        /**
         * Returns the sub devices.
         *
         * @return the sub devices
         */
        public List<Device> getDevices() {
            return devices;
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
            if (!super.equals(o)) {
                return false;
            }
            MultiDevice that = (MultiDevice) o;
            return Objects.equals(devices, that.devices);
        }

        /** {@inheritDoc} */
        @Override
        public int hashCode() {
            return Objects.hash(super.hashCode(), devices);
        }

        /** {@inheritDoc} */
        @Override
        public String toString() {
            return deviceType + "()";
        }
    }
}
