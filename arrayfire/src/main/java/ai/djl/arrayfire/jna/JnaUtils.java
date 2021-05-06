/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.arrayfire.jna;

import ai.djl.Device;
import ai.djl.arrayfire.engine.AfDataType;
import ai.djl.arrayfire.engine.AfDeviceType;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;

/**
 * A class containing utilities to interact with the ArrayFire Engine's Java Native Access (JNA)
 * layer.
 */
@SuppressWarnings("MissingJavadocMethod")
public final class JnaUtils {
    private static final ArrayFireLibrary LIB = LibUtils.loadLibrary();

    private JnaUtils() {}

    // Error message list:
    // https://arrayfire.org/docs/defines_8h.htm#a82b94dc53bbd100a0e8ca9dd356aaf4f
    private static void checkCall(int error) {
        if (error != 0) {
            throw new EngineException("Failed with error code: " + error);
        }
    }

    public static void printInfo() {
        checkCall(LIB.af_info());
    }

    public static void setDevice(Device device) {
        // TODO: OpenCL is the default device
        int deviceInt = AfDeviceType.toDeviceType(device);
        checkCall(LIB.af_set_backend(deviceInt));
    }

    public static Pointer createNDArray(Buffer data, Shape shape, DataType dataType) {
        PointerByReference ptr = new PointerByReference();
        Pointer dataPtr = Native.getDirectBufferPointer(data);
        long[] shapeLong = new long[4];
        Arrays.fill(shapeLong, 1);
        for (int i = 0; i < shape.dimension(); i++) {
            shapeLong[i] = shape.get(i);
        }
        checkCall(
                LIB.af_create_array(
                        ptr,
                        dataPtr,
                        shape.dimension(),
                        shapeLong,
                        AfDataType.toArrayFire(dataType)));
        return ptr.getValue();
    }

    public static Shape getShape(Pointer handle) {
        IntBuffer lengthHolder = IntBuffer.allocate(1);
        checkCall(LIB.af_get_numdims(lengthHolder, handle));
        int length = lengthHolder.get();
        LongBuffer[] buf = new LongBuffer[4];
        for (int i = 0; i < 4; i++) {
            buf[i] = LongBuffer.allocate(1);
        }
        checkCall(LIB.af_get_dims(buf[0], buf[1], buf[2], buf[3], handle));
        long[] shape = new long[length];
        for (int i = 0; i < length; i++) {
            shape[i] = buf[i].get();
        }
        return new Shape(shape);
    }

    public static void getByteBuffer(ByteBuffer buf, Pointer handle) {
        Pointer pointer = Native.getDirectBufferPointer(buf);
        checkCall(LIB.af_get_data_ptr(pointer, handle));
    }

    public static void releaseNDArray(Pointer handle) {
        checkCall(LIB.af_release_array(handle));
    }

    public static int getRefCounts(Pointer handle) {
        IntBuffer buf = IntBuffer.allocate(1);
        checkCall(LIB.af_get_data_ref_count(buf, handle));
        return buf.get();
    }
}
