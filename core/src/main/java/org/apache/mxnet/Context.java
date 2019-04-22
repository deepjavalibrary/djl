/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package org.apache.mxnet;

import org.apache.mxnet.jna.JnaUtils;

public class Context {

    private DeviceType deviceType;
    private int deviceId;
    private static Context defaultContext;

    static {
        if (JnaUtils.getGpuCount() > 0) {
            defaultContext = gpu();
        } else {
            defaultContext = cpu();
        }
    }

    public Context(DeviceType deviceType, int deviceId) {
        this.deviceType = deviceType;
        this.deviceId = deviceId;
    }

    public DeviceType getDeviceType() {
        return deviceType;
    }

    public int getDeviceId() {
        return deviceId;
    }

    @Override
    public String toString() {
        return deviceType.name() + '(' + deviceId + ')';
    }

    public static Context cpu() {
        return cpu(0);
    }

    public static Context cpu(int deviceId) {
        return new Context(DeviceType.CPU, deviceId);
    }

    public static Context gpu() {
        return gpu(0);
    }

    public static Context gpu(int deviceId) {
        return new Context(DeviceType.GPU, deviceId);
    }

    public static Context getDefaultContext() {
        return defaultContext;
    }

    public enum DeviceType {
        CPU(1),
        GPU(2),
        CPU_PINNED(3);

        private int type;

        DeviceType(int type) {
            this.type = type;
        }

        public int getType() {
            return type;
        }

        public static DeviceType fromValue(int value) {
            for (DeviceType type : values()) {
                if (type.type == value) {
                    return type;
                }
            }
            throw new IllegalArgumentException("Invalid DeviceType value: " + value);
        }
    }
}
