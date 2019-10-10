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

import software.amazon.ai.Device;

public final class DeviceType {

    private static final String CPU_PINNED = "cpu_pinned";

    private DeviceType() {}

    public static int toDeviceType(Device device) {
        String deviceType = device.getDeviceType();

        if (Device.cpu().getDeviceType().equals(deviceType)) {
            return 1;
        } else if (Device.gpu().getDeviceType().equals(deviceType)) {
            return 2;
        } else if (CPU_PINNED.equals(deviceType)) {
            return 3;
        } else {
            throw new IllegalArgumentException("Unsupported device: " + device.toString());
        }
    }

    public static String fromDeviceType(int deviceType) {
        switch (deviceType) {
            case 1:
                return Device.cpu().getDeviceType();
            case 2:
                return Device.gpu().getDeviceType();
            case 3:
                return "cpu_pinned";
            default:
                throw new IllegalArgumentException("Unsupported deviceType: " + deviceType);
        }
    }
}
