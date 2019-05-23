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
package com.amazon.ai;

import com.amazon.ai.engine.Engine;
import java.util.Objects;

public class Context {

    private static final Context CPU = new Context("CPU", 0);
    private static final Context GPU = new Context("GPU", 0);

    private String deviceType;
    private int deviceId;

    public Context(String deviceType, int deviceId) {
        this.deviceType = deviceType;
        this.deviceId = deviceId;
    }

    public String getDeviceType() {
        return deviceType;
    }

    public int getDeviceId() {
        return deviceId;
    }

    @Override
    public String toString() {
        return deviceType + '(' + deviceId + ')';
    }

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

    @Override
    public int hashCode() {
        return Objects.hash(deviceType, deviceId);
    }

    public static Context cpu() {
        return CPU;
    }

    public static Context cpu(int deviceId) {
        return new Context("CPU", deviceId);
    }

    public static Context gpu() {
        return GPU;
    }

    public static Context gpu(int deviceId) {
        return new Context("GPU", deviceId);
    }

    public static Context defaultContext() {
        return Engine.getInstance().defaultContext();
    }
}
