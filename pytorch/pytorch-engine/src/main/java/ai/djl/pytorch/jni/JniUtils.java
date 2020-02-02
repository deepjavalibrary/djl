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
package ai.djl.pytorch.jni;

import ai.djl.Device;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtDeviceType;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class JniUtils {
    private JniUtils() {}

    public static DataType getDataType(Pointer ndArray) {
        int dataType = PyTorchLibrary.LIB.torchDType(ndArray);
        return DataType.values()[dataType];
    }

    public static Device getDevice(Pointer ndArray) {
        int[] device = PyTorchLibrary.LIB.torchDevice(ndArray);
        return new Device(PtDeviceType.fromDeviceType(device[0]), device[1]);
    }

    public static Shape getShape(Pointer handle) {
        return new Shape(PyTorchLibrary.LIB.torchSizes(handle));
    }

    public static ByteBuffer getByteBuffer(Pointer handle) {
        ByteBuffer bb = PyTorchLibrary.LIB.torchDataPtr(handle);
        bb.order(ByteOrder.nativeOrder());
        return bb;
    }
}
