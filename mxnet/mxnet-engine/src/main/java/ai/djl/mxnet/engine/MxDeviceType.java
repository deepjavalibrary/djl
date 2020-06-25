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
import ai.djl.DeviceType;

/** DeviceType is the MXNet equivalent of the types in {@link Device}. */
public final class MxDeviceType implements DeviceType {

    private static final String CPU_PINNED = "cpu_pinned";

    private MxDeviceType() {}

    /**
     * Converts a {@link Device} to the corresponding MXNet device number.
     *
     * @param device the java {@link Device}
     * @return the MXNet device number
     */
    public static int toDeviceType(Device device) {
        String deviceType = device.getDeviceType();

        if (Device.Type.CPU.equals(deviceType)) {
            return 1;
        } else if (Device.Type.GPU.equals(deviceType)) {
            return 2;
        } else if (CPU_PINNED.equals(deviceType)) {
            return 3;
        } else {
            throw new IllegalArgumentException("Unsupported device: " + device.toString());
        }
    }

    /**
     * Converts from an MXNet device number to {@link Device}.
     *
     * @param deviceType the MXNet device number
     * @return the corresponding {@link Device}
     */
    public static String fromDeviceType(int deviceType) {
        switch (deviceType) {
            case 1:
            case 3:
                // hide the CPU_PINNED to frontend user
                // but the advance user can still create CPU_PINNED
                // to pass through engine
                return Device.Type.CPU;
            case 2:
                return Device.Type.GPU;
            default:
                throw new IllegalArgumentException("Unsupported deviceType: " + deviceType);
        }
    }
}
