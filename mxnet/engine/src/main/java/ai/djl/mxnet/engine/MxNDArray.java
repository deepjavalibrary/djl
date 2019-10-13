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
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.ndarray.Matrix;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.index.NDIndexFullSlice;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.internal.NDFormat;
import ai.djl.ndarray.types.DataDesc;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.training.GradReq;
import ai.djl.util.Utils;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.Stack;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class MxNDArray extends NativeResource implements NDArray {

    private Device device;
    private SparseFormat sparseFormat;
    private DataType dataType;
    private Shape shape;
    private MxNDManager manager;
    private MxNDArrayEx mxNDArrayEx;

    MxNDArray(MxNDManager manager, Pointer handle, Device device, Shape shape, DataType dataType) {
        this(manager, handle);
        this.device = device;
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
    public Device getDevice() {
        if (device == null) {
            device = JnaUtils.getDevice(getHandle());
        }
        return device;
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
        manager.attach(getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detach(getUid());
        manager = MxNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asInDevice(Device dev, boolean copy) {
        if (dev.equals(getDevice()) && !copy) {
            return slice();
        }
        MxNDArray nd = manager.create(getShape(), getDataType(), dev);
        copyTo(nd);
        return nd;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asType(DataType dtype, boolean copy) {
        if (dtype.equals(getDataType()) && !copy) {
            return slice();
        }
        MxNDArray nd = manager.create(getShape(), dtype, getDevice());
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

    /**
     * Computes the gradients of the NDArray w.r.t variables.
     *
     * @param retainGraph Whether to retain the computation graph for another backward pass on the
     *     same graph. By default the computation history is cleared.
     */
    public void backward(boolean retainGraph) {

        JnaUtils.autogradBackward(new NDList(this), retainGraph ? 1 : 0);
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient() {
        attachGradient(GradReq.WRITE, null);
    }

    void attachGradient(GradReq gradReq, SparseFormat sparseFormat) {
        // Does zerosLike support sparse?
        MxNDArray grad;
        if (sparseFormat == null || sparseFormat == SparseFormat.UNDEFINED) {
            grad = (MxNDArray) zerosLike();
        } else {
            grad = (MxNDArray) manager.zeros(getShape(), getDataType(), getDevice());
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
        if (pointer == null) {
            throw new IllegalStateException(
                    "No gradient attached to this NDArray, please call array.attachGradient()"
                            + "on your NDArray or block.setInitializer() on your Block");
        }
        return manager.create(pointer);
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        Shape sh = getShape();
        DataType dType = getDataType();
        long product = sh.size();
        long len = dType.getNumOfBytes() * product;
        ByteBuffer bb = manager.allocateDirect(Math.toIntExact(len));
        Pointer pointer = Native.getDirectBufferPointer(bb);
        JnaUtils.syncCopyToCPU(getHandle(), pointer, Math.toIntExact(product));
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
        ByteBuffer buf = manager.allocateDirect(size * numOfBytes);

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
    public void set(NDIndex index, NDArray value) {
        NDIndexFullSlice fullSlice = index.getAsFullSlice(getShape()).orElse(null);
        if (fullSlice != null) {
            MxOpParams params = new MxOpParams();
            params.addTupleParam("begin", fullSlice.getMin());
            params.addTupleParam("end", fullSlice.getMax());
            params.addTupleParam("step", fullSlice.getStep());

            Stack<NDArray> prepareValue = new Stack<>();
            prepareValue.add(value);
            prepareValue.add(prepareValue.peek().asInDevice(getDevice(), false));
            // prepareValue.add(prepareValue.peek().asType(getDataType(), false));
            // Deal with the case target: (1, 10, 1), original (10)
            // try to find (10, 1) and reshape (10) to that
            Shape targetShape = fullSlice.getShape();
            while (targetShape.size() > value.size()) {
                targetShape = targetShape.slice(1);
            }
            prepareValue.add(prepareValue.peek().reshape(targetShape));
            prepareValue.add(prepareValue.peek().broadcast(fullSlice.getShape()));

            manager.invoke(
                    "_npi_slice_assign",
                    new NDList(this, prepareValue.peek()),
                    new NDList(this),
                    params);
            for (NDArray toClean : prepareValue) {
                if (toClean != value) {
                    toClean.close();
                }
            }
            return;
        }
        throw new UnsupportedOperationException(
                "set() currently supports all, fixed, and slices indices");
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, Number value) {
        NDIndexFullSlice fullSlice = index.getAsFullSlice(getShape()).orElse(null);
        if (fullSlice != null) {
            MxOpParams params = new MxOpParams();
            params.addTupleParam("begin", fullSlice.getMin());
            params.addTupleParam("end", fullSlice.getMax());
            params.addTupleParam("step", fullSlice.getStep());
            params.addParam("scalar", value);
            manager.invoke("_npi_slice_assign_scalar", new NDList(this), new NDList(this), params);
            return;
        }
        throw new UnsupportedOperationException(
                "set() currently supports all, fixed, and slices indices");
    }

    /** {@inheritDoc} */
    @Override
    public void setScalar(NDIndex index, Number value) {
        NDIndexFullSlice fullSlice = index.getAsFullSlice(getShape()).orElse(null);
        if (fullSlice != null) {
            if (fullSlice.getShape().size() != 1) {
                throw new IllegalArgumentException("The provided index does not set a scalar");
            }
            set(index, value);
            return;
        }
        throw new UnsupportedOperationException(
                "set() currently supports all, fixed, and slices indices");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDIndex index) {
        if (index.getRank() == 0 && getShape().isScalar()) {
            // TODO: return a slice once MXNet support it
            return duplicate();
        }

        NDIndexFullSlice fullSlice = index.getAsFullSlice(getShape()).orElse(null);
        if (fullSlice != null) {
            MxOpParams params = new MxOpParams();
            params.addTupleParam("begin", fullSlice.getMin());
            params.addTupleParam("end", fullSlice.getMax());
            params.addTupleParam("step", fullSlice.getStep());
            NDArray result = manager.invoke("_npi_slice", this, params);
            if (!fullSlice.getToSqueeze().isEmpty()) {
                NDArray oldResult = result;
                result =
                        result.squeeze(
                                fullSlice.getToSqueeze().stream().mapToInt(i -> i).toArray());
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
                    "shape are diff. Required: " + destShape + ", Actual " + inShape);
        }
        NDList src = new NDList(this);
        NDList dest = new NDList(ndArray);
        manager.invoke("_npi_copyto", src, dest, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike() {
        return manager.invoke("_np_zeros_like", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        return manager.invoke("_np_ones_like", this, null);
    }

    @Override
    public boolean contentEquals(Number number) {
        if (number == null) {
            return false;
        }
        try (NDArray result = eq(number)) {
            return result.nonzero() == result.size();
        }
    }

    @Override
    public boolean contentEquals(NDArray other) {
        if (other == null || (!getShape().equals(other.getShape()))) {
            return false;
        }
        if (getDataType() != other.getDataType()) {
            return false;
        }
        try (NDArray result = eq(other)) {
            return result.nonzero() == result.size();
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean allClose(NDArray other, float rtol, float atol, boolean equalNan) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return manager.invoke("_npi_equal_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        return manager.invoke("_npi_equal", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return manager.invoke("_npi_not_equal_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        return manager.invoke("_npi_not_equal", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return manager.invoke("_npi_greater_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        return manager.invoke("_npi_greater", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return manager.invoke("_npi_greater_equal_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        return manager.invoke("_npi_greater_equal", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return manager.invoke("_npi_less_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        return manager.invoke("_npi_less", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return manager.invoke("_npi_less_equal_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        return manager.invoke("_npi_less_equal", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray where(NDArray condition, NDArray other) {
        NDArray array1;
        NDArray array2;
        if (!getShape().equals(other.getShape())) {
            Shape res = deriveBroadcastedShape(getShape(), other.getShape());
            array1 = (!res.equals(getShape())) ? broadcast(res) : this;
            array2 = (!res.equals(other.getShape())) ? other.broadcast(res) : other;
        } else {
            array1 = this;
            array2 = other;
        }
        try {
            return manager.invoke("where", new NDList(condition, array1, array2), null).head();
        } finally {
            if (array1 != this) {
                array1.close();
            }
            if (array2 != other) {
                array2.close();
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_npi_add_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray... others) {
        NDList toAdd = new NDList(this);
        toAdd.addAll(new NDList(others));
        if (others.length == 0) {
            throw new IllegalArgumentException("Passed in arrays must have at least one element");
        } else if (others.length == 1) {
            return manager.invoke("_npi_add", toAdd, null).head();
        } else {
            return manager.invoke("add_n", toAdd, null).head();
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_npi_subtract_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return manager.invoke("_npi_subtract", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_npi_multiply_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray... others) {
        if (others == null || others.length == 0) {
            throw new IllegalArgumentException("Passed in arrays must have at least one element");
        }
        NDArray current = this;
        for (NDArray other : others) {
            NDArray next = manager.invoke("_npi_multiply", new NDList(current, other), null).head();
            if (current != this) {
                current.close();
            }
            current = next;
        }
        return current;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toSparse(SparseFormat fmt) {
        if (fmt == SparseFormat.DENSE) {
            throw new IllegalArgumentException("Default type is not allowed");
        }
        if (fmt == getSparseFormat()) {
            return slice();
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
        return manager.invoke("_npi_true_divide_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return manager.invoke("_npi_true_divide", new NDList(this, other), null).head();
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
        return manager.invoke("_npi_power_scalar", this, params);
    }

    @Override
    public NDArray pow(NDArray other) {
        return manager.invoke("_npi_power", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        manager.invoke("_npi_add_scalar", new NDList(this), new NDList(this), params);
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
            manager.invoke("_npi_add", toAdd, new NDList(this), null);
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
        manager.invoke("_npi_subtract_scalar", new NDList(this), new NDList(this), params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        manager.invoke("_npi_subtract", new NDList(this, other), new NDList(this), null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray... others) {
        if (others == null || others.length == 0) {
            throw new IllegalArgumentException("Passed in arrays must have at least one element");
        }
        for (NDArray other : others) {
            manager.invoke("_npi_multiply", new NDList(this, other), new NDList(this), null);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        manager.invoke("_npi_multiply_scalar", new NDList(this), new NDList(this), params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        manager.invoke("_npi_true_divide_scalar", new NDList(this), new NDList(this), params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        manager.invoke("_npi_true_divide", new NDList(this, other), new NDList(this), null);
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
        manager.invoke("_npi_power_scalar", new NDList(this), new NDList(this), params);
        return this;
    }

    @Override
    public NDArray powi(NDArray other) {
        manager.invoke("_npi_power", new NDList(this, other), new NDList(this), null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        return manager.invoke("_npi_negative", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        manager.invoke("_npi_negative", new NDList(this), new NDList(this), null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        return manager.invoke("_npi_absolute", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return manager.invoke("_npi_square", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        return manager.invoke("_npi_cbrt", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        return manager.invoke("_npi_floor", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        return manager.invoke("_npi_ceil", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        return manager.invoke("round", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        return manager.invoke("_npi_trunc", this, null);
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
        return manager.invoke("_npi_tan", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        return manager.invoke("_npi_arcsin", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        return manager.invoke("_npi_arccos", this, null);
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
        return manager.invoke("_npi_tanh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        return manager.invoke("_npi_arcsinh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        return manager.invoke("_npi_arccosh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        return manager.invoke("_npi_arctanh", this, null);
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
    public NDArray max() {
        return manager.invoke("_np_max", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        return manager.invoke("_np_max", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return manager.invoke("_np_max", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min() {
        return manager.invoke("_np_min", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return manager.invoke("_np_min", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum() {
        return manager.invoke("_np_sum", this, null);
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
    public NDArray prod() {
        return manager.invoke("_np_prod", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return manager.invoke("_np_prod", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        return manager.invoke("_npi_mean", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return manager.invoke("_npi_mean", this, params);
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
        MxOpParams params = new MxOpParams();
        params.addParam("newshape", shape);
        return manager.invoke("_np_reshape", this, params);
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
    public NDArray squeeze() {
        return manager.invoke("_np_squeeze", this, null);
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
    public NDArray stack(NDList arrays, int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        NDArray[] srcArray = new NDArray[arrays.size() + 1];
        srcArray[0] = this;
        System.arraycopy(arrays.toArray(), 0, srcArray, 1, arrays.size());
        return manager.invoke("_npi_stack", new NDList(srcArray), params).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray concat(NDList list, int axis) {
        NDArray[] arrays = list.toArray();
        if (Stream.of(arrays).allMatch(array -> array.getShape().dimension() == 0)) {
            throw new IllegalArgumentException(
                    "scalar(zero-dimensional) arrays cannot be concatenated");
        }
        int dimension = arrays[0].getShape().dimension();
        for (int i = 1; i < arrays.length; i++) {
            if (arrays[i].getShape().dimension() != dimension) {
                throw new IllegalArgumentException(
                        "all the input arrays must have same number of dimensions, but the array at index 0 has "
                                + dimension
                                + " dimension(s) and the array at index "
                                + i
                                + " has "
                                + arrays[i].getShape().dimension()
                                + " dimension(s)");
            }
        }
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
    public NDArray logicalAnd(NDArray other) {
        // TODO switch to numpy op, although current op support zero-dim, scalar
        return manager.invoke("broadcast_logical_and", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray other) {
        // TODO switch to numpy op, although current op support zero-dim, scalar
        return manager.invoke("broadcast_logical_or", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray other) {
        // TODO switch to numpy op, although current op support zero-dim, scalar
        return manager.invoke("broadcast_logical_xor", new NDList(this, other), null).head();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        return manager.invoke("_npi_logical_not", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argsort(int axis, boolean ascending) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        // be careful that MXNet numpy argsort op didn't officially support this param
        params.addParam("is_ascend", ascending);
        params.setDataType(DataType.INT32);
        return manager.invoke("argsort", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        // TODO remove scalar, zero-dim check once MXNet support it
        if (isEmpty() || isScalar()) {
            long dim = getShape().dimension();
            if (axis >= dim) {
                throw new IllegalArgumentException(
                        "axis " + axis + "is out of bounds for array of dimension " + dim);
            }
            return duplicate();
        }
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return manager.invoke("sort", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        if (isEmpty() || isScalar()) {
            return duplicate();
        }
        return manager.invoke("sort", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes) {
        return softmax(axes, 1.0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes, double temperature) {
        // TODO remove this after MXNet softmax fix zero-dim issue
        if (isEmpty()) {
            return getManager().create(getShape());
        }
        MxOpParams params = new MxOpParams();
        if (axes.length != 1) {
            long size = shape.size(axes);
            NDArray transposed = transpose(axes);
            Shape transposedShape = transposed.getShape();
            Shape sliced = transposed.getShape().slice(axes.length);
            NDArray array = transposed.reshape(new Shape(size).addAll(sliced));
            params.addParam("axis", 0);
            params.addParam("temperature", temperature);
            return manager.invoke("_npx_softmax", array, params)
                    .reshape(transposedShape)
                    .transpose(axes);
        } else {
            params.addParam("axis", axes[0]);
            params.addParam("temperature", temperature);
            return manager.invoke("_npx_softmax", this, params);
        }
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
            return slice();
        }
        return castStorage(SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long repeats) {
        // zero-dim
        if (isEmpty()) {
            return duplicate();
        }
        // scalar
        int dim = (isScalar()) ? 1 : getShape().dimension();
        long[] repeatsArray = new long[dim];
        Arrays.fill(repeatsArray, repeats);
        return tile(repeatsArray);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, long repeats) {
        // scalar
        if (isScalar()) {
            throw new IllegalArgumentException("scalar didn't support specifying axis");
        }
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
        // zero-dim
        if (isEmpty()) {
            return duplicate();
        }
        // scalar
        int dim = (isScalar()) ? 1 : getShape().dimension();
        long[] repeatsArray = new long[dim];
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
        // TODO get rid of for loop once bug in MXNet np.repeat is fixed
        NDArray array = this;
        int baseAxis = getShape().dimension() - repeats.length;
        for (int i = 0; i < repeats.length; i++) {
            if (repeats[i] > 1) {
                NDArray previousArray = array;
                MxOpParams params = new MxOpParams();
                params.addParam("repeats", repeats[i]);
                params.addParam("axis", baseAxis + i);
                array = manager.invoke("_np_repeat", array, params);
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
    public NDArray dot(NDArray other) {
        return manager.invoke("_np_dot", new NDList(this, other), null).head();
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
        return manager.invoke("_np_transpose", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int... dimensions) {
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
        return manager.invoke("_np_transpose", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        MxOpParams params = new MxOpParams();
        params.setShape(shape);
        return manager.invoke("_np_broadcast_to", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argmax() {
        return manager.invoke("_npi_argmax", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argmax(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return manager.invoke("_npi_argmax", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argmin() {
        // TODO switch to MXNet numpy argmin
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argmin of an empty NDArray");
        }
        NDArray array = (isScalar()) ? reshape(1) : this;
        try (NDArray temp = manager.invoke("argmin", array, null)) {
            return temp.reshape(new Shape());
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argmin(int axis) {
        // TODO switch to MXNet numpy argmin
        NDArray array = (isScalar()) ? reshape(1) : this;
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        NDArray temp = manager.invoke("argmin", array, params);
        if (isScalar()) {
            NDArray res = temp.reshape(new Shape());
            temp.close();
            return res;
        }
        return temp;
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
        // TODO switch to use nonzero op in mxnet
        if (isEmpty()) {
            return 0;
        }
        MxNDArray zeros = (MxNDArray) eq(0);
        NDArray sum = manager.invoke("_np_sum", eq(zeros).eq(zeros), null);
        return sum.toArray()[0].longValue();
    }

    /** {@inheritDoc} */
    @Override
    public boolean isEmpty() {
        return getShape().size() == 0;
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayEx getNDArrayInternal() {
        return mxNDArrayEx;
    }

    private long[] repeatsToMatchShape(Shape desiredShape) {
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
            if (curShape.get(i) == 0 || desiredShape.get(i) % curShape.get(i) != 0) {
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

    // TODO only used to calculate zero-dim numpy shape
    // remove it once MXNet have all the np op that we support
    private Shape deriveBroadcastedShape(Shape lhs, Shape rhs) {
        long[] result = new long[Math.max(lhs.dimension(), rhs.dimension())];
        long lDiff = result.length - lhs.dimension();
        long rDiff = result.length - rhs.dimension();
        for (int i = 0; i < result.length; i++) {
            long l = 1;
            long r = 1;
            if (i >= lDiff) {
                l = lhs.get(Math.toIntExact(i - lDiff));
            }
            if (i >= rDiff) {
                r = rhs.get(Math.toIntExact(i - rDiff));
            }
            if (l != r) {
                if (l != 1 && r != 1) {
                    throw new IllegalArgumentException(
                            "operands could not be broadcast together with shapes "
                                    + lhs
                                    + " "
                                    + rhs);
                }
                result[i] = (l == 1) ? r : l;
            } else {
                result[i] = l;
            }
        }
        return new Shape(result);
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

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof MxNDArray) {
            return contentEquals((MxNDArray) obj);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (isReleased()) {
            return "This array is already closed";
        }
        if (Utils.DEBUG) {
            return NDFormat.format(this);
        }
        return super.toString();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            // TODO: remove after fixing multi-thread data loading issue
            // JnaUtils.waitToRead(pointer);
            JnaUtils.freeNdArray(pointer);
            manager.detach(getUid());
            manager = null;
        }
    }
}
