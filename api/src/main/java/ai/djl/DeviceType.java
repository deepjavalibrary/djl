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

/** {@code DeviceType} is a class used to map the Device name to their corresponding type number. */
public interface DeviceType {

    /**
     * Map device to its type number.
     *
     * @param device {@link Device} to map from
     * @return the number specified by engine
     */
    static int toDeviceType(Device device) {
        return 0;
    }

    /**
     * Map device to its type number.
     *
     * @param deviceType the number specified by engine
     * @return {@link Device} to map to
     */
    static String fromDeviceType(int deviceType) {
        return null;
    }
}
