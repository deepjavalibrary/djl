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
package org.apache.mxnet.engine;

import com.amazon.ai.util.PairList;
import com.amazon.ai.util.Utils;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Map;
import org.apache.mxnet.Context;
import org.apache.mxnet.jna.FunctionInfo;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.jna.PointerArray;
import org.apache.mxnet.types.DataType;
import org.apache.mxnet.types.StorageType;

public class NdArray extends NativeResource {

    private static final Map<String, FunctionInfo> OPS = JnaUtils.getNdArrayFunctions();

    private static final int MAX_DEPTH = 10;
    private static final int MAX_PRINT_ROWS = 10;
    private static final int MAX_PRINT_ITEMS = 20;
    private static final String LF = System.getProperty("line.separator");

    private Context context;
    private StorageType storageType;
    private DataType dataType;
    private Shape shape;

    public NdArray(ResourceAllocator alloc, Pointer handle) {
        super(alloc, handle);
    }

    public NdArray(
            ResourceAllocator alloc,
            Context context,
            StorageType storageType,
            Shape shape,
            DataType dataType,
            Pointer handle) {
        super(alloc, handle);
        this.context = context;
        this.dataType = dataType;
        this.shape = shape;
        this.storageType = storageType;
    }

    public NdArray(Context context, Shape shape) {
        this(null, context, shape, DataType.FLOAT32, false);
    }

    public NdArray(ResourceAllocator alloc, Context context, Shape shape) {
        this(alloc, context, shape, DataType.FLOAT32, false);
    }

    public NdArray(
            ResourceAllocator alloc,
            Context context,
            Shape shape,
            DataType dataType,
            boolean delay) {
        this(
                alloc,
                context,
                null,
                shape,
                DataType.FLOAT32,
                JnaUtils.createNdArray(context, shape, dataType, shape.dimension(), delay));
    }

    public DataType getDataType() {
        if (dataType == null) {
            dataType = JnaUtils.getDataType(getHandle());
        }
        return dataType;
    }

    public Context getContext() {
        if (context == null) {
            context = JnaUtils.getContext(getHandle());
        }
        return context;
    }

    public Shape getShape() {
        if (shape == null) {
            shape = JnaUtils.getShape(getHandle());
        }
        return shape;
    }

    public StorageType getStorageType() {
        if (storageType == null) {
            storageType = JnaUtils.getStorageType(getHandle());
        }
        return storageType;
    }

    public void set(Buffer data) {
        if (data.remaining() != shape.product()) {
            throw new IllegalArgumentException(
                    "array size ("
                            + data.remaining()
                            + ")do not match the size of NDArray ("
                            + shape.product());
        }
        JnaUtils.syncCopyFromCPU(getHandle(), data);
    }

    public NdArray at(int index) {
        Pointer pointer = JnaUtils.ndArrayAt(getHandle(), index);
        return new NdArray(alloc, pointer);
    }

    public NdArray slice(int begin, int end) {
        Pointer pointer = JnaUtils.slice(getHandle(), begin, end);
        return new NdArray(alloc, pointer);
    }

    public void copyTo(NdArray ndArray) {
        Shape inShape = getShape();
        Shape destShape = ndArray.getShape();
        if (!Arrays.equals(inShape.getShape(), destShape.getShape())) {
            throw new IllegalArgumentException("shape are diff.");
        }

        FunctionInfo functionInfo = OPS.get("_copyto");

        PointerByReference ref = new PointerByReference(new PointerArray(ndArray.getHandle()));
        functionInfo.invoke(getHandle(), ref, null);
    }

    public void waitToRead() {
        JnaUtils.waitToRead(getHandle());
    }

    public void waitToWrite() {
        JnaUtils.waitToWrite(getHandle());
    }

    public void waitAll() {
        JnaUtils.waitAll();
    }

    public NdArray argsort(int axis, boolean isAscend) {
        FunctionInfo functionInfo = OPS.get("argsort");
        String[] keys = new String[] {"axis", "is_ascend"};
        String[] values = new String[2];
        values[0] = String.valueOf(axis);
        values[1] = isAscend ? "True" : "False";
        PairList<String, String> params = new PairList<>(keys, values);
        PointerByReference ref = new PointerByReference();

        functionInfo.invoke(getHandle(), ref, params);

        return new NdArray(alloc, ref.getValue().getPointerArray(0, 1)[0]);
    }

    public FunctionInfo genericNDArrayFunctionInvoke(String opName, Map<String, Object> args) {
        FunctionInfo func = OPS.get(opName);
        if (func == null) {
            throw new UnsupportedOperationException("Unsupported operation: " + opName);
        }

        return func;
    }

    public float[] toFloatArray() {
        FloatBuffer fb = toByteBuffer().asFloatBuffer();
        float[] ret = new float[fb.remaining()];
        fb.get(ret);
        return ret;
    }

    public byte[] toByteArray() {
        ByteBuffer bb = toByteBuffer();
        byte[] buf = new byte[bb.remaining()];
        bb.get(buf);
        return buf;
    }

    private ByteBuffer toByteBuffer() {
        Shape sh = getShape();
        DataType dType = getDataType();
        int product = sh.product();
        int len = dType.getNumOfBytes() * product;
        ByteBuffer bb = ByteBuffer.allocateDirect(len);
        Pointer pointer = Native.getDirectBufferPointer(bb);
        JnaUtils.syncCopyToCPU(getHandle(), pointer, product);
        bb.order(ByteOrder.LITTLE_ENDIAN);
        return bb;
    }

    private void dump(StringBuilder sb, int depth) {
        Utils.pad(sb, ' ', depth);
        sb.append('[');
        int len = getShape().head();
        if (getShape().dimension() == 1) {
            float[] arr = toFloatArray();
            int limit = Math.min(arr.length, MAX_PRINT_ITEMS);
            for (int i = 0; i < limit; ++i) {
                if (i > 0) {
                    sb.append(", ");
                }
                switch (getDataType()) {
                    case FLOAT32:
                    case FLOAT16:
                    case FLOAT64:
                        sb.append(String.format("%.8e", arr[i]));
                        break;
                    default:
                        sb.append((long) arr[i]);
                        break;
                }
            }
            int remaining = arr.length - limit;
            if (remaining > 0) {
                sb.append(", ... ").append(remaining).append(" more");
            }
        } else {
            sb.append(LF);
            int limit = Math.min(len, MAX_PRINT_ROWS);
            for (int i = 0; i < limit; ++i) {
                try (NdArray nd = at(i)) {
                    nd.dump(sb, depth + 1);
                }
            }
            int remaining = len - limit;
            if (remaining > 0) {
                Utils.pad(sb, ' ', depth + 1);
                sb.append("... ").append(remaining).append(" more");
            }
            Utils.pad(sb, ' ', depth);
        }
        sb.append("],").append(LF);
    }

    public String dump() {
        StringBuilder sb = new StringBuilder(200);
        sb.append("ND: ")
                .append(getShape())
                .append(' ')
                .append(getContext())
                .append(' ')
                .append(getDataType())
                .append(LF);
        if (getShape().dimension() < MAX_DEPTH) {
            dump(sb, 0);
        } else {
            sb.append("[ Exceed max print dimension ]");
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        if (Utils.DEBUG) {
            return dump();
        }
        return super.toString();
    }

    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            JnaUtils.freeNdArray(pointer);
        }
        if (alloc != null) {
            alloc.detach(this);
        }
    }
}
