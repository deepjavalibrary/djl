/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite;

import java.lang.reflect.Array;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.Arrays;

// TODO(b/153882978): Add scalar getters similar to TF's Java API.
@SuppressWarnings("MissingJavadocMethod")
public final class Tensor {

    static Tensor fromIndex(long nativeInterpreterHandle, int tensorIndex) {
        return new Tensor(create(nativeInterpreterHandle, tensorIndex));
    }

    public static class QuantizationParams {

        private final float scale;

        private final int zeroPoint;

        public QuantizationParams(final float scale, final int zeroPoint) {
            this.scale = scale;
            this.zeroPoint = zeroPoint;
        }

        public float getScale() {
            return scale;
        }

        public int getZeroPoint() {
            return zeroPoint;
        }
    }

    public void close() {
        delete(nativeHandle);
        nativeHandle = 0;
    }

    public DataType dataType() {
        return dtype;
    }

    public int numDimensions() {
        return shapeCopy.length;
    }

    public int numBytes() {
        return numBytes(nativeHandle);
    }

    public int numElements() {
        return computeNumElements(shapeCopy);
    }

    public int[] shape() {
        return shapeCopy;
    }

    public int[] shapeSignature() {
        return shapeSignatureCopy;
    }

    public int index() {
        return index(nativeHandle);
    }

    public String name() {
        return name(nativeHandle);
    }

    public QuantizationParams quantizationParams() {
        return quantizationParamsCopy;
    }

    public ByteBuffer asReadOnlyBuffer() {
        // Note that the ByteBuffer order is not preserved when duplicated or marked read only, so
        // we have to repeat the call.
        return buffer().asReadOnlyBuffer().order(ByteOrder.nativeOrder());
    }

    void setTo(Object src) {
        if (src == null) {
            if (hasDelegateBufferHandle(nativeHandle)) {
                return;
            }
            throw new IllegalArgumentException(
                    "Null inputs are allowed only if the Tensor is bound to a buffer handle.");
        }
        throwIfTypeIsIncompatible(src);
        throwIfSrcShapeIsIncompatible(src);
        if (isBuffer(src)) {
            setTo((Buffer) src);
        } else if (dtype == DataType.STRING && shapeCopy.length == 0) {
            // Update scalar string input with 1-d byte array.
            writeScalar(nativeHandle, src);
        } else if (src.getClass().isArray()) {
            writeMultiDimensionalArray(nativeHandle, src);
        } else {
            writeScalar(nativeHandle, src);
        }
    }

    private void setTo(Buffer src) {
        // Note that we attempt to use a direct memcpy optimization for direct, native-ordered
        // buffers.
        // There are no base Buffer#order() or Buffer#put() methods, so again we have to ugly cast.
        if (src instanceof ByteBuffer) {
            ByteBuffer srcBuffer = (ByteBuffer) src;
            if (srcBuffer.isDirect() && srcBuffer.order() == ByteOrder.nativeOrder()) {
                writeDirectBuffer(nativeHandle, src);
            } else {
                buffer().put(srcBuffer);
            }
        } else if (src instanceof LongBuffer) {
            LongBuffer srcBuffer = (LongBuffer) src;
            if (srcBuffer.isDirect() && srcBuffer.order() == ByteOrder.nativeOrder()) {
                writeDirectBuffer(nativeHandle, src);
            } else {
                buffer().asLongBuffer().put(srcBuffer);
            }
        } else if (src instanceof FloatBuffer) {
            FloatBuffer srcBuffer = (FloatBuffer) src;
            if (srcBuffer.isDirect() && srcBuffer.order() == ByteOrder.nativeOrder()) {
                writeDirectBuffer(nativeHandle, src);
            } else {
                buffer().asFloatBuffer().put(srcBuffer);
            }
        } else if (src instanceof IntBuffer) {
            IntBuffer srcBuffer = (IntBuffer) src;
            if (srcBuffer.isDirect() && srcBuffer.order() == ByteOrder.nativeOrder()) {
                writeDirectBuffer(nativeHandle, src);
            } else {
                buffer().asIntBuffer().put(srcBuffer);
            }
        } else if (src instanceof ShortBuffer) {
            ShortBuffer srcBuffer = (ShortBuffer) src;
            if (srcBuffer.isDirect() && srcBuffer.order() == ByteOrder.nativeOrder()) {
                writeDirectBuffer(nativeHandle, src);
            } else {
                buffer().asShortBuffer().put(srcBuffer);
            }
        } else {
            throw new IllegalArgumentException("Unexpected input buffer type: " + src);
        }
    }

    Object copyTo(Object dst) {
        if (dst == null) {
            if (hasDelegateBufferHandle(nativeHandle)) {
                return dst;
            }
            throw new IllegalArgumentException(
                    "Null outputs are allowed only if the Tensor is bound to a buffer handle.");
        }
        throwIfTypeIsIncompatible(dst);
        throwIfDstShapeIsIncompatible(dst);
        if (isBuffer(dst)) {
            copyTo((Buffer) dst);
        } else {
            readMultiDimensionalArray(nativeHandle, dst);
        }
        return dst;
    }

    private void copyTo(Buffer dst) {
        // There is no base Buffer#put() method, so we have to ugly cast.
        if (dst instanceof ByteBuffer) {
            ((ByteBuffer) dst).put(buffer());
        } else if (dst instanceof FloatBuffer) {
            ((FloatBuffer) dst).put(buffer().asFloatBuffer());
        } else if (dst instanceof LongBuffer) {
            ((LongBuffer) dst).put(buffer().asLongBuffer());
        } else if (dst instanceof IntBuffer) {
            ((IntBuffer) dst).put(buffer().asIntBuffer());
        } else if (dst instanceof ShortBuffer) {
            ((ShortBuffer) dst).put(buffer().asShortBuffer());
        } else {
            throw new IllegalArgumentException("Unexpected output buffer type: " + dst);
        }
    }

    // TODO(b/80431971): Remove this method after deprecating multi-dimensional array inputs.
    int[] getInputShapeIfDifferent(Object input) {
        if (input == null) {
            return null;
        }
        // Implicit resizes based on ByteBuffer capacity isn't supported, so short-circuit that
        // path.
        // The Buffer's size will be validated against this Tensor's size in {@link #setTo(Object)}.
        if (isBuffer(input)) {
            return null;
        }
        throwIfTypeIsIncompatible(input);
        int[] inputShape = computeShapeOf(input);
        if (Arrays.equals(shapeCopy, inputShape)) {
            return null;
        }
        return inputShape;
    }

    void refreshShape() {
        this.shapeCopy = shape(nativeHandle);
    }

    DataType dataTypeOf(Object o) {
        if (o != null) {
            Class<?> c = o.getClass();
            // For arrays, the data elements must be a *primitive* type, e.g., an
            // array of floats is fine, but not an array of Floats.
            if (c.isArray()) {
                while (c.isArray()) {
                    c = c.getComponentType();
                }
                if (float.class.equals(c)) {
                    return DataType.FLOAT32;
                } else if (int.class.equals(c)) {
                    return DataType.INT32;
                } else if (short.class.equals(c)) {
                    return DataType.INT16;
                } else if (byte.class.equals(c)) {
                    // Byte array can be used for storing string tensors, especially for
                    // ParseExample op.
                    if (dtype == DataType.STRING) {
                        return DataType.STRING;
                    }
                    return DataType.UINT8;
                } else if (long.class.equals(c)) {
                    return DataType.INT64;
                } else if (boolean.class.equals(c)) {
                    return DataType.BOOL;
                } else if (String.class.equals(c)) {
                    return DataType.STRING;
                }
            } else {
                // For scalars, the type will be boxed.
                if (Float.class.equals(c) || o instanceof FloatBuffer) {
                    return DataType.FLOAT32;
                } else if (Integer.class.equals(c) || o instanceof IntBuffer) {
                    return DataType.INT32;
                } else if (Short.class.equals(c) || o instanceof ShortBuffer) {
                    return DataType.INT16;
                } else if (Byte.class.equals(c)) {
                    // Note that we don't check for ByteBuffer here; ByteBuffer payloads
                    // are allowed to map to any type, and should be handled earlier
                    // in the input/output processing pipeline.
                    return DataType.UINT8;
                } else if (Long.class.equals(c) || o instanceof LongBuffer) {
                    return DataType.INT64;
                } else if (Boolean.class.equals(c)) {
                    return DataType.BOOL;
                } else if (String.class.equals(c)) {
                    return DataType.STRING;
                }
            }
        } else {
            throw new NullPointerException();
        }
        throw new IllegalArgumentException(
                "DataType error: cannot resolve DataType of " + o.getClass().getName());
    }

    int[] computeShapeOf(Object o) {
        int size = computeNumDimensions(o);
        if (dtype == DataType.STRING) {
            Class<?> c = o.getClass();
            if (c.isArray()) {
                while (c.isArray()) {
                    c = c.getComponentType();
                }
                // If the given string data is stored in byte streams, the last array dimension
                // should be
                // treated as a value.
                if (byte.class.equals(c)) {
                    --size;
                }
            }
        }
        int[] dimensions = new int[size];
        fillShape(o, 0, dimensions);
        return dimensions;
    }

    static int computeNumElements(int[] shape) {
        int n = 1;
        for (int i = 0; i < shape.length; ++i) {
            n *= shape[i];
        }
        return n;
    }

    static int computeNumDimensions(Object o) {
        if (o == null || !o.getClass().isArray()) {
            return 0;
        }
        if (Array.getLength(o) == 0) {
            throw new IllegalArgumentException("Array lengths cannot be 0.");
        }
        return 1 + computeNumDimensions(Array.get(o, 0));
    }

    static void fillShape(Object o, int dim, int[] shape) {
        if (shape == null || dim == shape.length) {
            return;
        }
        final int len = Array.getLength(o);
        if (shape[dim] == 0) {
            shape[dim] = len;
        } else if (shape[dim] != len) {
            throw new IllegalArgumentException(
                    String.format(
                            "Mismatched lengths (%d and %d) in dimension %d",
                            shape[dim], len, dim));
        }
        for (int i = 0; i < len; ++i) {
            fillShape(Array.get(o, i), dim + 1, shape);
        }
    }

    private void throwIfTypeIsIncompatible(Object o) {
        // ByteBuffer payloads can map to any type, so exempt it from the check.
        if (isByteBuffer(o)) {
            return;
        }
        DataType oType = dataTypeOf(o);

        if (oType != dtype) {
            // INT8 and UINT8 have the same string name, "byte"
            if (oType.toStringName().equals(dtype.toStringName())) {
                return;
            }

            throw new IllegalArgumentException(
                    String.format(
                            "Cannot convert between a TensorFlowLite tensor with type %s and a Java"
                                + " object of type %s (which is compatible with the TensorFlowLite"
                                + " type %s).",
                            dtype, o.getClass().getName(), oType));
        }
    }

    private void throwIfSrcShapeIsIncompatible(Object src) {
        if (isBuffer(src)) {
            Buffer srcBuffer = (Buffer) src;
            int bytes = numBytes();
            // Note that we allow the client to provide a ByteBuffer even for non-byte Tensors.
            // In such cases, we only care that the raw byte capacity matches the tensor byte
            // capacity.
            int srcBytes =
                    isByteBuffer(src)
                            ? srcBuffer.capacity()
                            : srcBuffer.capacity() * dtype.byteSize();
            if (bytes != srcBytes) {
                throw new IllegalArgumentException(
                        String.format(
                                "Cannot copy to a TensorFlowLite tensor (%s) with %d bytes from a "
                                        + "Java Buffer with %d bytes.",
                                name(), bytes, srcBytes));
            }
            return;
        }
        int[] srcShape = computeShapeOf(src);
        if (!Arrays.equals(srcShape, shapeCopy)) {
            throw new IllegalArgumentException(
                    String.format(
                            "Cannot copy to a TensorFlowLite tensor (%s) with shape %s from a Java"
                                    + " object with shape %s.",
                            name(), Arrays.toString(shapeCopy), Arrays.toString(srcShape)));
        }
    }

    private void throwIfDstShapeIsIncompatible(Object dst) {
        if (isBuffer(dst)) {
            Buffer dstBuffer = (Buffer) dst;
            int bytes = numBytes();
            // Note that we allow the client to provide a ByteBuffer even for non-byte Tensors.
            // In such cases, we only care that the raw byte capacity fits the tensor byte capacity.
            // This is subtly different than Buffer *inputs*, where the size should be exact.
            int dstBytes =
                    isByteBuffer(dst)
                            ? dstBuffer.capacity()
                            : dstBuffer.capacity() * dtype.byteSize();
            if (bytes > dstBytes) {
                throw new IllegalArgumentException(
                        String.format(
                                "Cannot copy from a TensorFlowLite tensor (%s) with %d bytes to a "
                                        + "Java Buffer with %d bytes.",
                                name(), bytes, dstBytes));
            }
            return;
        }
        int[] dstShape = computeShapeOf(dst);
        if (!Arrays.equals(dstShape, shapeCopy)) {
            throw new IllegalArgumentException(
                    String.format(
                            "Cannot copy from a TensorFlowLite tensor (%s) with shape %s to a Java"
                                    + " object with shape %s.",
                            name(), Arrays.toString(shapeCopy), Arrays.toString(dstShape)));
        }
    }

    private static boolean isBuffer(Object o) {
        return o instanceof Buffer;
    }

    private static boolean isByteBuffer(Object o) {
        return o instanceof ByteBuffer;
    }

    private long nativeHandle;
    private final DataType dtype;
    private int[] shapeCopy;
    private final int[] shapeSignatureCopy;
    private final QuantizationParams quantizationParamsCopy;

    private Tensor(long nativeHandle) {
        this.nativeHandle = nativeHandle;
        this.dtype = DataType.fromC(dtype(nativeHandle));
        this.shapeCopy = shape(nativeHandle);
        this.shapeSignatureCopy = shapeSignature(nativeHandle);
        this.quantizationParamsCopy =
                new QuantizationParams(
                        quantizationScale(nativeHandle), quantizationZeroPoint(nativeHandle));
    }

    public ByteBuffer buffer() {
        return buffer(nativeHandle).order(ByteOrder.nativeOrder());
    }

    private static native long create(long interpreterHandle, int tensorIndex);

    private static native void delete(long handle);

    private static native ByteBuffer buffer(long handle);

    private static native void writeDirectBuffer(long handle, Buffer src);

    private static native int dtype(long handle);

    private static native int[] shape(long handle);

    private static native int[] shapeSignature(long handle);

    private static native int numBytes(long handle);

    private static native boolean hasDelegateBufferHandle(long handle);

    private static native void readMultiDimensionalArray(long handle, Object dst);

    private static native void writeMultiDimensionalArray(long handle, Object src);

    private static native void writeScalar(long handle, Object src);

    private static native int index(long handle);

    private static native String name(long handle);

    private static native float quantizationScale(long handle);

    private static native int quantizationZeroPoint(long handle);
}
