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

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import org.apache.mxnet.jna.JnaUtils;
import software.amazon.ai.Context;
import software.amazon.ai.ndarray.Matrix;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.index.NDIndex;
import software.amazon.ai.ndarray.index.NDIndexAll;
import software.amazon.ai.ndarray.index.NDIndexElement;
import software.amazon.ai.ndarray.index.NDIndexFixed;
import software.amazon.ai.ndarray.index.NDIndexSlice;
import software.amazon.ai.ndarray.internal.NDArrayEx;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.ndarray.types.SparseFormat;
import software.amazon.ai.training.GradReq;
import software.amazon.ai.util.Utils;

public class MxNDArray extends NativeResource implements NDArray {

    private static final int MAX_DEPTH = 10;
    private static final int MAX_PRINT_ROWS = 10;
    private static final int MAX_PRINT_ITEMS = 20;
    private static final String LF = System.getProperty("line.separator");

    private Context context;
    private SparseFormat sparseFormat;
    private DataType dataType;
    private Shape shape;
    private MxNDManager manager;
    private MxNDArrayEx mxNDArrayEx;

    MxNDArray(
            MxNDManager manager, Pointer handle, Context context, Shape shape, DataType dataType) {
        this(manager, handle);
        this.context = context;
        // shape check
        if (Arrays.stream(shape.getShape()).anyMatch(s -> s < 0)) {
            throw new IllegalArgumentException("The shape must be >= 0");
        }
        this.shape = shape;
        this.dataType = dataType;
    }

    MxNDArray(MxNDManager manager, Pointer handle) {
        super(handle);
        this.manager = manager;
        this.mxNDArrayEx = new MxNDArrayEx(this);
    }

    MxNDArray(MxNDManager manager, Pointer handle, SparseFormat fmt) {
        this(manager, handle);
        this.sparseFormat = fmt;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        if (dataType == null) {
            dataType = JnaUtils.getDataType(getHandle());
        }
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Context getContext() {
        if (context == null) {
            context = JnaUtils.getContext(getHandle());
        }
        return context;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape == null) {
            shape = JnaUtils.getShape(getHandle());
        }
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        if (sparseFormat == null) {
            sparseFormat = JnaUtils.getStorageType(getHandle());
        }
        return sparseFormat;
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc getDataDescriptor() {
        return new DataDesc(getShape(), getDataType(), null);
    }

    /** {@inheritDoc} */
    @Override
    public void attach(NDManager manager) {
        detach();
        this.manager = (MxNDManager) manager;
        manager.attach(this);
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detach(this);
        manager = MxNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray asInContext(Context ctx, boolean copy) {
        if (ctx.equals(getContext()) && !copy) {
            return this;
        }
        MxNDArray nd = manager.create(getShape(), getDataType(), ctx);
        copyTo(nd);
        return nd;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray asType(DataType dtype, boolean copy) {
        if (dtype.equals(getDataType()) && !copy) {
            return this;
        }
        MxNDArray nd = manager.create(getShape(), dtype, getContext());
        copyTo(nd);
        return nd;
    }

    /** {@inheritDoc} */
    @Override
    public Matrix asMatrix() {
        if (!shape.isMatrix()) {
            throw new IllegalStateException("NDArray is not a matrix");
        }
        return new MxMatrix(this);
    }

    /** {@inheritDoc} */
    @Override
    public void backward() {
        backward(null, false, true);
    }

    /** {@inheritDoc} */
    @Override
    public void backward(boolean retainGraph, boolean isTraining) {
        backward(null, retainGraph, isTraining);
    }

    /** {@inheritDoc} */
    @Override
    public void backward(NDArray outGrad, boolean retainGraph, boolean isTraining) {
        Pointer outGradHandle;
        if (outGrad != null) {
            MxNDArray outGradND = (MxNDArray) outGrad;
            outGradHandle = outGradND.getHandle();
        } else {
            outGradHandle = null;
        }

        JnaUtils.autogradBackwardExecute(
                1,
                getHandle(),
                outGradHandle,
                0,
                null,
                retainGraph ? 1 : 0,
                0,
                isTraining ? 1 : 0,
                null,
                null);
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient() {
        attachGradient(GradReq.WRITE, null);
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient(GradReq gradReq, SparseFormat sparseFormat) {
        // Does zerosLike support sparse?
        MxNDArray grad;
        if (sparseFormat == null || sparseFormat == SparseFormat.UNDEFINED) {
            grad = (MxNDArray) zerosLike();
        } else {
            grad = (MxNDArray) manager.zeros(getShape(), getDataType(), getContext());
        }
        int gradReqValue = gradReq.getValue();
        IntBuffer gradReqBuffer = IntBuffer.allocate(1);
        gradReqBuffer.put(0, gradReqValue);
        JnaUtils.autogradMarkVariables(1, getHandle(), gradReqBuffer, grad.getHandle());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        Pointer pointer = JnaUtils.getGradient(getHandle());
        return manager.create(pointer);
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        Shape sh = getShape();
        DataType dType = getDataType();
        long product = sh.size();
        long len = dType.getNumOfBytes() * product;
        ByteBuffer bb = ByteBuffer.allocateDirect(Math.toIntExact(len));
        Pointer pointer = Native.getDirectBufferPointer(bb);
        JnaUtils.syncCopyToCPU(getHandle(), pointer, Math.toIntExact(product));
        bb.order(ByteOrder.LITTLE_ENDIAN);
        return bb;
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        int size = data.remaining();
        DataType inputType = DataType.fromBuffer(data);
        validate(inputType, size);

        if (data.isDirect()) {
            JnaUtils.syncCopyFromCPU(getHandle(), data, size);
            return;
        }

        int numOfBytes = inputType.getNumOfBytes();
        ByteBuffer buf = ByteBuffer.allocateDirect(size * numOfBytes);
        buf.order(ByteOrder.LITTLE_ENDIAN); // MXNet use little endian

        switch (inputType) {
            case FLOAT32:
                buf.asFloatBuffer().put((FloatBuffer) data);
                break;
            case FLOAT64:
                buf.asDoubleBuffer().put((DoubleBuffer) data);
                break;
            case UINT8:
            case INT8:
                buf.put((ByteBuffer) data);
                break;
            case INT32:
                buf.asIntBuffer().put((IntBuffer) data);
                break;
            case INT64:
                buf.asLongBuffer().put((LongBuffer) data);
                break;
            case FLOAT16:
            default:
                throw new AssertionError("Show never happen");
        }
        JnaUtils.syncCopyFromCPU(getHandle(), buf, size);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray set(NDIndex index, NDArray value) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray set(NDIndex index, Number value) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray setElement(NDIndex index, Number value) throws IllegalArgumentException {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray seti(NDIndex index, NDArray value) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray seti(NDIndex index, Number value) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray setElementi(NDIndex index, Number value) throws IllegalArgumentException {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDIndex index) {
        if (index.getRank() == 0) {
            return this;
        }

        if (index.stream()
                .allMatch(
                        ie ->
                                ie instanceof NDIndexAll
                                        || ie instanceof NDIndexFixed
                                        || ie instanceof NDIndexSlice)) {
            int dimensions = index.getRank();
            if (dimensions > getShape().dimension()) {
                throw new IllegalArgumentException("The index has too many dimensions");
            }
            long[] min = new long[dimensions];
            long[] max = new long[dimensions];
            long[] step = new long[dimensions];
            List<Integer> toSqueeze = new ArrayList<>(dimensions);
            for (int i = 0; i < dimensions; i++) {
                NDIndexElement ie = index.get(i);
                if (ie instanceof NDIndexFixed) {
                    min[i] = ((NDIndexFixed) ie).getIndex();
                    max[i] = ((NDIndexFixed) ie).getIndex() + 1;
                    step[i] = 1;
                    toSqueeze.add(i);
                } else if (ie instanceof NDIndexSlice) {
                    NDIndexSlice slice = (NDIndexSlice) ie;
                    min[i] = Optional.ofNullable(slice.getMin()).orElse(0L);
                    max[i] = Optional.ofNullable(slice.getMax()).orElse(size(i));
                    step[i] = Optional.ofNullable(slice.getStep()).orElse(1L);
                } else if (ie instanceof NDIndexAll) {
                    min[i] = 0;
                    max[i] = size(i);
                    step[i] = 1;
                }
            }
            MxOpParams params = new MxOpParams();
            params.addTupleParam("begin", min);
            params.addTupleParam("end", max);
            params.addTupleParam("step", step);
            NDArray result = manager.invoke("slice", this, params);
            if (!toSqueeze.isEmpty()) {
                NDArray oldResult = result;
                result = result.squeeze(toSqueeze.stream().mapToInt(i -> i).toArray());
                oldResult.close();
            }
            return result;
        }
        throw new UnsupportedOperationException(
                "get() currently supports all, fixed, and slices indices");
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray ndArray) {
        if (!(ndArray instanceof MxNDArray)) {
            throw new IllegalArgumentException("Only MxNDArray is supported.");
        }
        Shape inShape = getShape();
        Shape destShape = ndArray.getShape();
        if (!Arrays.equals(inShape.getShape(), destShape.getShape())) {
            throw new IllegalArgumentException(
                    String.format("shape are diff. Required: %s, Actual %s", destShape, inShape));
        }
        NDList src = new NDList(this);
        NDList dest = new NDList(ndArray);
        manager.invoke("_copyto", src, dest, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dup() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike() {
        return manager.invoke("zeros_like", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        return manager.invoke("ones_like", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray like() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public boolean contentEquals(Number number) {
        try (NDArray result = eq(number)) {
            return result.nonzero() == result.size();
        }
    }

    @Override
    public boolean contentEquals(NDArray other) {
        try (NDArray result = eq(other)) {
            return result.nonzero() == result.size();
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean equalsWithEps(Object o, double eps) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(Number other) {
        try (NDArray numbers = zerosLike()) {
            numbers.addi(other);
            return eq(numbers);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        return manager.invoke("_equal", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eps(Number other) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eps(NDArray other) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number other) {
        return eq(other).eq(0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        return eq(other).eq(0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return manager.invoke("_greater_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        return manager.invoke("_greater", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return manager.invoke("_greater_equal_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        return manager.invoke("_greater_equal", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return manager.invoke("_lesser_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        return manager.invoke("_lesser", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return manager.invoke("_lesser_equal_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        return manager.invoke("_lesser_equal", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_plus_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray... others) {
        NDList toAdd = new NDList(this);
        toAdd.addAll(new NDList(others));
        if (others.length == 0) {
            throw new IllegalArgumentException("Passed in arrays must have at least one element");
        } else if (others.length == 1) {
            return manager.invoke("_plus", toAdd, null).head();
        } else {
            return manager.invoke("add_n", toAdd, null).head();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_minus_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return manager.invoke("_sub", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_mul_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray... others) {
        if (others == null || others.length == 0) {
            throw new IllegalArgumentException("Passed in arrays must have at least one element");
        }
        NDArray current = this;
        for (NDArray other : others) {
            NDArray next = manager.invoke("_mul", new NDList(current, other), null).head();
            if (current != this) {
                current.close();
            }
            current = next;
        }
        return current;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSparse() {
        return getSparseFormat() != SparseFormat.DENSE;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toSparse(SparseFormat fmt) {
        if (fmt == SparseFormat.DENSE) {
            throw new IllegalArgumentException("Default type is not allowed");
        }
        if (fmt == getSparseFormat()) {
            return this;
        }
        return castStorage(fmt);
    }

    private NDArray castStorage(SparseFormat fmt) {
        MxOpParams params = new MxOpParams();
        params.setParam("stype", fmt.getType());
        return manager.invoke("cast_storage", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_div_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return manager.invoke("elemwise_div", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_npi_mod_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        return manager.invoke("_npi_mod", new NDList(this, other), null).head();
    }

    @Override
    public NDArray pow(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_power_scalar", this, params);
    }

    @Override
    public NDArray pow(NDArray other) {
        return manager.invoke("_power", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        manager.invoke("_plus_scalar", new NDList(this), new NDList(this), params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray... others) {
        NDList toAdd = new NDList(this);
        toAdd.addAll(new NDList(others));
        if (others.length == 0) {
            throw new IllegalArgumentException("Passed in arrays must have at least one element");
        } else if (others.length == 1) {
            manager.invoke("_plus", toAdd, new NDList(this), null);
        } else {
            manager.invoke("add_n", toAdd, new NDList(this), null);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        manager.invoke("_minus_scalar", new NDList(this), new NDList(this), params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        manager.invoke("_sub", new NDList(this, other), new NDList(this), null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray... others) {
        if (others == null || others.length == 0) {
            throw new IllegalArgumentException("Passed in arrays must have at least one element");
        }
        for (NDArray other : others) {
            manager.invoke("_mul", new NDList(this, other), new NDList(this), null);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        manager.invoke("_mul_scalar", new NDList(this), new NDList(this), params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        manager.invoke("_div_scalar", new NDList(this), new NDList(this), params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        manager.invoke("elemwise_div", new NDList(this, other), new NDList(this), null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        manager.invoke("_npi_mod_scalar", new NDList(this), new NDList(this), params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        manager.invoke("_npi_mod", new NDList(this, other), new NDList(this), null);
        return this;
    }

    @Override
    public NDArray powi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        manager.invoke("_power_scalar", new NDList(this), new NDList(this), params);
        return this;
    }

    @Override
    public NDArray powi(NDArray other) {
        manager.invoke("_power", new NDList(this, other), new NDList(this), null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        return manager.invoke("negative", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        manager.invoke("negative", new NDList(this), new NDList(this), null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        return manager.invoke("_np_absolute", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return manager.invoke("_npi_square", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        return manager.invoke("_np_cbrt", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        return manager.invoke("_np_floor", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        return manager.invoke("_np_ceil", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        return manager.invoke("round", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        return manager.invoke("_np_trunc", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        return manager.invoke("_npi_exp", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        return manager.invoke("_npi_log", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        return manager.invoke("_npi_log10", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        return manager.invoke("_npi_log2", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        return manager.invoke("_npi_sin", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        return manager.invoke("_npi_cos", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        return manager.invoke("_np_tan", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        return manager.invoke("_npi_arcsin", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        return manager.invoke("_np_arccos", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        return manager.invoke("_npi_arctan", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        return manager.invoke("_npi_sinh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        return manager.invoke("_npi_cosh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return manager.invoke("_np_tanh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        return manager.invoke("_np_arcsinh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        return manager.invoke("_np_arccosh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        return manager.invoke("_np_arctanh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        return manager.invoke("_npi_degrees", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        return manager.invoke("_npi_radians", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public Number max() {
        return manager.invoke("max", this, null).toArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        return manager.invoke("max", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return manager.invoke("max", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public Number min() {
        MxOpParams params = new MxOpParams();
        return manager.invoke("min", this, params).toArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return manager.invoke("min", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public Number sum() {
        MxOpParams params = new MxOpParams();
        return manager.invoke("_np_sum", this, params).toArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return manager.invoke("_np_sum", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public Number prod() {
        MxOpParams params = new MxOpParams();
        return manager.invoke("prod", this, params).toArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return manager.invoke("prod", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public Number mean() {
        MxOpParams params = new MxOpParams();
        return manager.invoke("_np_mean", this, params).toArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return manager.invoke("_np_mean", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis1", axis1);
        params.addParam("axis2", axis2);
        return manager.invoke("_np_trace", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int[] indices, int axis) {
        MxOpParams params = new MxOpParams();
        // follow the numpy behavior
        if (indices[0] != 0) {
            int[] tempIndices = new int[indices.length + 1];
            tempIndices[0] = 0;
            System.arraycopy(indices, 0, tempIndices, 1, indices.length);
            indices = tempIndices;
        }
        params.addTupleParam("indices", indices);
        params.addParam("axis", axis);
        params.addParam("squeeze_axis", false);
        return manager.invoke("_npi_split", new NDList(this), params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        return reshape(new Shape(Math.toIntExact(size())));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        long count = shape.getUnknownValueCount();
        if (count == 0 && getShape().size() != shape.size()) {
            throw new IllegalArgumentException("The given shape does not match the current shape");
        }
        // check multiply -1 in shape
        if (count > 1) {
            throw new IllegalArgumentException(
                    "Shape could only have a single unknown dimension (-1).");
        }
        // check size outside the -1 is divisible factor of the original size
        long sizeWithoutUnknown =
                Arrays.stream(shape.getShape()).filter(s -> s != -1).reduce(1, Math::multiplyExact);
        if (getShape().size() % sizeWithoutUnknown != 0) {
            throw new IllegalArgumentException("The given shape does not match the current shape");
        }
        Pointer pointer = JnaUtils.reshape(getHandle(), shape.getShape(), false);
        return manager.create(pointer);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return manager.invoke("_npi_expand_dims", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        return manager.invoke("_np_squeeze", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stack(NDArray[] arrays, int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        NDArray[] srcArray = new NDArray[arrays.length + 1];
        srcArray[0] = this;
        System.arraycopy(arrays, 0, srcArray, 1, arrays.length);
        return manager.invoke("_npi_stack", new NDList(srcArray), params).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stack(NDList arrays, int axis) {
        return stack(arrays.toArray(), axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray concat(NDArray[] arrays, int axis) {
        MxOpParams params = new MxOpParams();
        // MXNet backend use dim as argument name
        params.addParam("dim", axis);
        NDArray[] srcArray = new NDArray[arrays.length + 1];
        srcArray[0] = this;
        System.arraycopy(arrays, 0, srcArray, 1, arrays.length);
        return manager.invoke("_npi_concatenate", new NDList(srcArray), params).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argsort(int axis, boolean ascending) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        params.addParam("is_ascend", ascending);
        params.setDataType(DataType.INT32);
        return manager.invoke("argsort", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return manager.invoke("sort", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        MxOpParams params = new MxOpParams();
        return manager.invoke("sort", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes) {
        if (axes.length != 1) {
            // TODO:
            throw new UnsupportedOperationException("Not implemented");
        }
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axes[0]);
        return manager.invoke("softmax", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes, double temperature) {
        if (axes.length != 1) {
            // TODO:
            throw new UnsupportedOperationException("Not implemented");
        }
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axes[0]);
        params.addParam("temperature", temperature);
        return manager.invoke("softmax", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsumi(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        params.setDataType(getDataType());
        manager.invoke("_np_cumsum", new NDList(this), new NDList(this), params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsumi() {
        manager.invoke("_np_cumsum", new NDList(this), new NDList(this), null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsum(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        params.setDataType(getDataType());
        return manager.invoke("_np_cumsum", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsum() {
        return manager.invoke("_np_cumsum", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createMask(NDIndex index) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createMask(Predicate<Number> predicate) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDense() {
        if (!isSparse()) {
            return this;
        }
        return castStorage(SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long repeats) {
        long[] repeatsArray = new long[getShape().dimension()];
        Arrays.fill(repeatsArray, repeats);
        return tile(repeatsArray);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, long repeats) {
        long[] repeatsArray = new long[getShape().dimension()];
        Arrays.fill(repeatsArray, 1);
        repeatsArray[withAxis(axis)] = repeats;
        return tile(repeatsArray);
    }

    @Override
    public NDArray tile(long[] repeats) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("reps", repeats);
        return manager.invoke("_npi_tile", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(Shape desiredShape) {
        return tile(repeatsToMatchShape(desiredShape));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long repeats) {
        long[] repeatsArray = new long[getShape().dimension()];
        Arrays.fill(repeatsArray, repeats);
        return repeat(repeatsArray);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int axis, long repeats) {
        long[] repeatsArray = new long[getShape().dimension()];
        Arrays.fill(repeatsArray, 1);
        repeatsArray[withAxis(axis)] = repeats;
        return repeat(repeatsArray);
    }

    @Override
    public NDArray repeat(long[] repeats) {
        NDArray array = this;
        int baseAxis = getShape().dimension() - repeats.length;
        for (int i = 0; i < repeats.length; i++) {
            if (repeats[i] > 1) {
                NDArray previousArray = array;
                MxOpParams params = new MxOpParams();
                params.addParam("repeats", repeats[i]);
                params.addParam("axis", baseAxis + i);
                array = manager.invoke("repeat", array, params);
                if (previousArray != this) {
                    previousArray.close();
                }
            }
        }
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(Shape desiredShape) {
        return repeat(repeatsToMatchShape(desiredShape));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mmul(NDArray other) {
        return manager.invoke("dot", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(double min, double max) {
        MxOpParams params = new MxOpParams();
        params.addParam("a_min", min);
        params.addParam("a_max", max);
        return manager.invoke("_npi_clip", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray swapAxes(int axis1, int axis2) {
        MxOpParams params = new MxOpParams();
        params.addParam("dim1", axis1);
        params.addParam("dim2", axis2);
        return manager.invoke("_npi_swapaxes", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        MxOpParams params = new MxOpParams();
        return manager.invoke("transpose", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int[] dimensions) {
        if (Arrays.stream(dimensions).anyMatch((int d) -> d < 0)) {
            throw new UnsupportedOperationException(
                    "Passing -1 for broadcasting the dimension is not currently supported");
        }
        if (!Arrays.equals(
                Arrays.stream(dimensions).sorted().toArray(),
                IntStream.range(0, getShape().dimension()).toArray())) {
            throw new IllegalArgumentException(
                    "You must include each of the dimensions from 0 until "
                            + getShape().dimension());
        }
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axes", dimensions);
        return manager.invoke("transpose", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(long... shape) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(NDArray result) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public boolean equalShapes(NDArray other) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public NDArray argmax() {
        return manager.invoke("argmax", this, null);
    }

    @Override
    public NDArray argmax(int axis, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        params.addParam("keepdims", keepDims);
        return manager.invoke("argmax", this, params);
    }

    @Override
    public NDArray argmin() {
        return manager.invoke("argmin", this, null);
    }

    @Override
    public NDArray argmin(int axis, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        params.addParam("keepdims", keepDims);
        return manager.invoke("argmin", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public Number percentileNumber(Number percentile) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public Number medianNumber() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median(int... dimension) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int... dimension) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public long nonzero() {
        MxNDArray zeros = (MxNDArray) eq(0);
        NDArray sum = manager.invoke("sum", eq(zeros).eq(zeros), null);
        return sum.toArray()[0].intValue();
    }

    /** {@inheritDoc} */
    @Override
    public boolean isEmpty() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public boolean all() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public boolean any() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public boolean none() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        return manager.invoke("_np_logical_not", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayEx getNDArrayInternal() {
        return mxNDArrayEx;
    }

    private long[] repeatsToMatchShape(Shape desiredShape) throws IllegalArgumentException {
        Shape curShape = getShape();
        int dimension = curShape.dimension();
        if (desiredShape.dimension() > dimension) {
            throw new IllegalArgumentException("The desired shape has too many dimensions");
        }
        if (desiredShape.dimension() < dimension) {
            int additionalDimensions = dimension - desiredShape.dimension();
            desiredShape = curShape.slice(0, additionalDimensions).addAll(desiredShape);
        }
        long[] repeats = new long[dimension];
        for (int i = 0; i < dimension; i++) {
            if (desiredShape.get(i) % curShape.get(i) != 0) {
                throw new IllegalArgumentException(
                        "The desired shape is not a multiple of the original shape");
            }
            repeats[i] = Math.round(Math.ceil((double) desiredShape.get(i) / curShape.get(i)));
        }
        return repeats;
    }

    private int withAxis(int axis) {
        return Math.floorMod(axis, getShape().dimension());
    }

    private void validate(DataType inputType, int size) {
        if (getDataType() != inputType
                && (dataType != DataType.UINT8 || inputType != DataType.INT8)) {
            // Infer DataType from Buffer always return INI8, make this a special case that
            // allows set UNIT array with regular ByteBuffer.
            throw new IllegalStateException(
                    "DataType mismatch, required: " + dataType + ", actual: " + inputType);
        }
        if (size != getShape().size()) {
            throw new IllegalArgumentException(
                    "array size (" + size + ") do not match NDArray shape: " + shape);
        }
    }

    public void waitToRead() {
        JnaUtils.waitToRead(getHandle());
    }

    public void waitToWrite() {
        JnaUtils.waitToWrite(getHandle());
    }

    public void waitAll() {
        JnaUtils.waitToRead(getHandle());
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (Utils.DEBUG) {
            return dump();
        }
        return super.toString();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            JnaUtils.freeNdArray(pointer);
            detach();
            manager = null;
        }
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
        // corner case: 0 dimension
        if (size() == 0) {
            sb.append("[]");
            return sb.toString();
        }
        if (getShape().dimension() < MAX_DEPTH) {
            dump(sb, 0);
        } else {
            sb.append("[ Exceed max print dimension ]");
        }
        return sb.toString();
    }

    private void dump(StringBuilder sb, int depth) {
        Utils.pad(sb, ' ', depth);
        sb.append('[');
        long len = getShape().head();
        if (getShape().dimension() == 1) {
            long limit = Math.min(getShape().head(), MAX_PRINT_ITEMS);
            ByteBuffer buf = toByteBuffer().slice();
            buf.limit(Math.toIntExact(limit * getDataType().getNumOfBytes()));
            buf.order(ByteOrder.LITTLE_ENDIAN);
            sb.append(Utils.toCharSequence(buf, getDataType()));
            long remaining = getShape().head() - limit;
            if (remaining > 0) {
                sb.append(", ... ").append(remaining).append(" more");
            }
        } else {
            sb.append(LF);
            long limit = Math.min(len, MAX_PRINT_ROWS);
            for (int i = 0; i < limit; ++i) {
                try (MxNDArray nd = (MxNDArray) get(i)) {
                    nd.dump(sb, depth + 1);
                }
            }
            long remaining = len - limit;
            if (remaining > 0) {
                Utils.pad(sb, ' ', depth + 1);
                sb.append("... ").append(remaining).append(" more");
            }
            Utils.pad(sb, ' ', depth);
        }
        sb.append("],").append(LF);
    }
}
