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
import java.io.OutputStream;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import org.apache.mxnet.jna.JnaUtils;
import software.amazon.ai.Context;
import software.amazon.ai.ndarray.Matrix;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDFactory;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.index.NDIndex;
import software.amazon.ai.ndarray.index.NDIndexElement;
import software.amazon.ai.ndarray.index.NDIndexFixed;
import software.amazon.ai.ndarray.index.NDIndexSlice;
import software.amazon.ai.ndarray.internal.NDArrayEx;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Layout;
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
    private MxNDFactory factory;
    private MxNDArrayEx mxNDArrayEx;

    MxNDArray(
            MxNDFactory factory, Context context, Shape shape, DataType dataType, Pointer handle) {
        super(handle);
        this.factory = factory;
        this.context = context;
        this.dataType = dataType;
        this.shape = shape;
        this.mxNDArrayEx = new MxNDArrayEx(this);
    }

    MxNDArray(MxNDFactory factory, Context context, Shape shape, DataType dataType) {
        this(
                factory,
                context,
                shape,
                dataType,
                JnaUtils.createNdArray(context, shape, dataType, shape.dimension(), false));
    }

    /** {@inheritDoc} */
    @Override
    public byte[] getEncoded() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public void encode(OutputStream os) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public NDFactory getFactory() {
        return factory;
    }

    public void detach() {
        factory.detach(this);
        factory = MxNDFactory.SYSTEM_FACTORY;
    }

    public void attach(MxNDFactory factory) {
        detach();
        this.factory = factory;
        factory.attach(this);
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

    public SparseFormat getSparseFormat() {
        if (sparseFormat == null) {
            sparseFormat = JnaUtils.getStorageType(getHandle());
        }
        return sparseFormat;
    }

    /** {@inheritDoc} */
    @Override
    public Layout getLayout() {
        return Layout.UNDEFINED;
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc getDataDescriptor() {
        return new DataDesc(getShape(), getDataType(), null, getLayout(), getContext());
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        int size = data.remaining();
        DataType inputType;
        if (data instanceof FloatBuffer) {
            inputType = DataType.FLOAT32;
        } else if (data instanceof DoubleBuffer) {
            inputType = DataType.FLOAT64;
        } else if (data instanceof IntBuffer) {
            inputType = DataType.INT32;
        } else if (data instanceof LongBuffer) {
            inputType = DataType.INT64;
        } else if (data instanceof ByteBuffer) {
            inputType = DataType.INT8;
        } else {
            throw new IllegalArgumentException(
                    "Unsupported buffer type: " + data.getClass().getSimpleName());
        }
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
            case INT8:
                buf.put((ByteBuffer) data);
                break;
            case INT32:
                buf.asIntBuffer().put((IntBuffer) data);
                break;
            case INT64:
                buf.asLongBuffer().put((LongBuffer) data);
                break;
            case UINT8:
            case FLOAT16:
            default:
                throw new AssertionError("Show never happen");
        }
        JnaUtils.syncCopyFromCPU(getHandle(), buf, size);
    }

    /** {@inheritDoc} */
    @Override
    public void set(float[] data) {
        set(FloatBuffer.wrap(data));
    }

    /** {@inheritDoc} */
    @Override
    public void set(int[] data) {
        set(IntBuffer.wrap(data));
    }

    /** {@inheritDoc} */
    @Override
    public void set(double[] data) {
        set(DoubleBuffer.wrap(data));
    }

    /** {@inheritDoc} */
    @Override
    public void set(long[] data) {
        set(LongBuffer.wrap(data));
    }

    /** {@inheritDoc} */
    @Override
    public void set(byte[] data) {
        set(ByteBuffer.wrap(data));
    }

    private void validate(DataType inputType, int size) {
        if (getDataType() != inputType) {
            throw new IllegalStateException(
                    "DataType mismatch, required: " + dataType + ", actual: " + inputType);
        }
        if (size != getShape().size()) {
            throw new IllegalArgumentException(
                    "array size (" + size + ") do not match NDArray shape: " + shape);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDIndex index) {
        if (index.getRank() == 0) {
            return this;
        } else if (index.getRank() == 1) {
            NDIndexElement ie = index.get(0);
            if (ie instanceof NDIndexFixed) {
                Pointer pointer = JnaUtils.ndArrayAt(getHandle(), ((NDIndexFixed) ie).getIndex());
                return factory.create(pointer);
            } else if (ie instanceof NDIndexSlice) {
                NDIndexSlice slice = (NDIndexSlice) ie;
                int min = Optional.ofNullable(slice.getMin()).orElse(0);
                int max = Optional.ofNullable(slice.getMax()).orElse((int) size(0));
                if (slice.getStep() != null) {
                    throw new UnsupportedOperationException(
                            "Slicing currently does not support step");
                }
                Pointer pointer = JnaUtils.slice(getHandle(), min, max);
                return factory.create(pointer);
            } else {
                throw new UnsupportedOperationException(
                        "get currently supports Fixed and Slice indexes");
            }
        } else {
            throw new UnsupportedOperationException("get currently supports only rank 1");
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getElement(NDIndex index) throws IllegalArgumentException {
        NDArray value = get(index);
        if (value.size() != 1) {
            throw new IllegalArgumentException("The supplied Index does not produce an element");
        } else {
            return value;
        }
    }

    /** {@inheritDoc} */
    @Override
    public long getLong(NDIndex index) throws IllegalArgumentException {
        return getElement(index).toLongArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public double getDouble(NDIndex index) throws IllegalArgumentException {
        return getElement(index).toDoubleArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public float getFloat(NDIndex index) throws IllegalArgumentException {
        return getElement(index).toFloatArray()[0];
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
        NDArray[] src = new NDArray[] {this};
        NDArray[] dest = new NDArray[] {ndArray};
        factory.invoke("_copyto", src, dest, null);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray asInContext(Context ctx, boolean copy) {
        if (ctx.equals(getContext()) && !copy) {
            return this;
        }
        MxNDArray nd = factory.create(ctx, getShape(), getDataType());
        copyTo(nd);
        return nd;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray asType(DataType dtype, boolean copy) {
        if (dtype.equals(getDataType()) && !copy) {
            return this;
        }
        MxNDArray nd = factory.create(getContext(), getShape(), dtype);
        copyTo(nd);
        return nd;
    }

    /** {@inheritDoc} */
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
    public void attachGrad() {
        attachGrad(GradReq.WRITE, null);
    }

    /** {@inheritDoc} */
    @Override
    public void attachGrad(GradReq gradReq, SparseFormat sparseFormat) {
        // TODO: should we close grad?
        // Does zerosLike support sparse?
        MxNDArray grad;
        if (sparseFormat == null || sparseFormat == SparseFormat.UNDEFINED) {
            grad = (MxNDArray) zerosLike();
        } else {
            grad = (MxNDArray) factory.zeros(context, shape, dataType);
        }
        int gradReqValue = gradReq.getValue();
        IntBuffer gradReqBuffer = IntBuffer.allocate(1);
        gradReqBuffer.put(0, gradReqValue);
        JnaUtils.autogradMarkVariables(1, getHandle(), gradReqBuffer, grad.getHandle());
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
    public NDArray getGradient() {
        Pointer pointer = JnaUtils.getGradient(getHandle());
        return factory.create(pointer);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return factory.invoke("sort", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        MxOpParams params = new MxOpParams();
        return factory.invoke("sort", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argsort(int axis, boolean ascending) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        params.addParam("is_ascend", ascending);
        params.setDataType(DataType.INT32);
        return factory.invoke("argsort", this, params);
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
        return factory.invoke("softmax", this, params);
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
        return factory.invoke("softmax", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int axis, boolean squeezeAxis) {
        MxOpParams params = new MxOpParams();
        params.addParam("num_outputs", size(axis));
        params.addParam("axis", axis);
        params.addParam("squeeze_axis", squeezeAxis);
        return new NDList(factory.invoke("split", new NDArray[] {this}, params));
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int axis, int numOutputs) throws IllegalArgumentException {
        MxOpParams params = new MxOpParams();
        params.addParam("num_outputs", numOutputs);
        params.addParam("axis", axis);
        return new NDList(factory.invoke("split", new NDArray[] {this}, params));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return factory.invoke("_plus_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        factory.invoke("_plus_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        return factory.invoke("_plus", new NDArray[] {this, other}, null)[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        factory.invoke("_plus", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike() {
        return factory.invoke("zeros_like", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        return factory.invoke("ones_like", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSparse() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsumi(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        params.setDataType(getDataType());
        factory.invoke("_np_cumsum", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsumi() {
        factory.invoke("_np_cumsum", new NDArray[] {this}, new NDArray[] {this}, null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsum(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        params.setDataType(getDataType());
        return factory.invoke("_np_cumsum", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsum() {
        return factory.invoke("_np_cumsum", this, null);
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
    public NDArray eq(Number other) {
        try (NDArray numbers = zerosLike()) {
            numbers.addi(other);
            return eq(numbers);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        return factory.invoke("_equal", new NDArray[] {this, other}, null)[0];
    }

    @Override
    public boolean contentEquals(NDArray other) {
        try (NDArray result = eq(other)) {
            return result.nonzero() == result.size();
        }
    }

    @Override
    public boolean contentEquals(Number number) {
        try (NDArray result = eq(number)) {
            return result.nonzero() == result.size();
        }
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
    public NDArray gt(NDArray other) {
        return factory.invoke("_greater", new NDArray[] {this, other}, null)[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        return factory.invoke("_greater_equal", new NDArray[] {this, other}, null)[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return factory.invoke("_greater_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return factory.invoke("_greater_equal_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return factory.invoke("_lesser_equal_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number other) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", other.toString());
        return factory.invoke("_lesser_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        return factory.invoke("_lesser_equal", new NDArray[] {this, other}, null)[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        return factory.invoke("_lesser", new NDArray[] {this, other}, null)[0];
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
    public NDArray neg() {
        return factory.invoke("negative", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        factory.invoke("negative", new NDArray[] {this}, new NDArray[] {this}, null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return factory.invoke("_div_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        factory.invoke("_div_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return factory.invoke("_mul_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        factory.invoke("_mul_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return factory.invoke("_minus_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        factory.invoke("_minus_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return factory.invoke("_npi_mod_scalar", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        factory.invoke("_npi_mod_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        return factory.invoke("_npi_mod", new NDArray[] {this, other}, null)[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        factory.invoke("_npi_mod", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
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
    public NDArray tile(int repeats) {
        int[] repeatsArray = new int[getShape().dimension()];
        Arrays.fill(repeatsArray, repeats);
        return tile(repeatsArray);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, int repeats) {
        int[] repeatsArray = new int[getShape().dimension()];
        Arrays.fill(repeatsArray, 1);
        repeatsArray[withAxis(axis)] = repeats;
        return tile(repeatsArray);
    }

    @Override
    public NDArray tile(int[] repeats) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("reps", repeats);
        return factory.invoke("tile", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(Shape desiredShape) {
        return tile(repeatsToMatchShape(desiredShape));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int repeats) {
        int[] repeatsArray = new int[getShape().dimension()];
        Arrays.fill(repeatsArray, repeats);
        return repeat(repeatsArray);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int axis, int repeats) {
        int[] repeatsArray = new int[getShape().dimension()];
        Arrays.fill(repeatsArray, 1);
        repeatsArray[withAxis(axis)] = repeats;
        return repeat(repeatsArray);
    }

    @Override
    public NDArray repeat(int[] repeats) {
        NDArray array = this;
        int baseAxis = getShape().dimension() - repeats.length;
        for (int i = 0; i < repeats.length; i++) {
            if (repeats[i] > 1) {
                NDArray previousArray = array;
                MxOpParams params = new MxOpParams();
                params.addParam("repeats", repeats[i]);
                params.addParam("axis", baseAxis + i);
                array = factory.invoke("repeat", array, params);
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

    private int[] repeatsToMatchShape(Shape desiredShape) throws IllegalArgumentException {
        Shape curShape = getShape();
        int dimension = curShape.dimension();
        if (desiredShape.dimension() > dimension) {
            throw new IllegalArgumentException("The desired shape has too many dimensions");
        }
        if (desiredShape.dimension() < dimension) {
            int additionalDimensions = dimension - desiredShape.dimension();
            desiredShape = curShape.slice(0, additionalDimensions).addAll(desiredShape);
        }
        int[] repeats = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            if (desiredShape.get(i) % curShape.get(i) != 0) {
                throw new IllegalArgumentException(
                        "The desired shape is not a multiple of the original shape");
            }
            repeats[i] =
                    (int) Math.round(Math.ceil((double) desiredShape.get(i) / curShape.get(i)));
        }
        return repeats;
    }

    private int withAxis(int axis) {
        return Math.floorMod(axis, getShape().dimension());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mmul(NDArray other) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public float[] toFloatArray() {
        if (getDataType() != DataType.FLOAT32) {
            throw new IllegalStateException(
                    "DataType mismatch, Required float, Actual " + getDataType());
        }
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

    /** {@inheritDoc} */
    @Override
    public int[] toIntArray() {
        if (getDataType() != DataType.INT32) {
            throw new IllegalStateException(
                    "DataType mismatch, Required int" + " Actual " + getDataType());
        }
        IntBuffer ib = toByteBuffer().asIntBuffer();
        int[] ret = new int[ib.remaining()];
        ib.get(ret);
        return ret;
    }

    /** {@inheritDoc} */
    @Override
    public long[] toLongArray() {
        if (getDataType() != DataType.INT64) {
            throw new IllegalStateException(
                    "DataType mismatch, Required long" + " Actual " + getDataType());
        }
        LongBuffer lb = toByteBuffer().asLongBuffer();
        long[] ret = new long[lb.remaining()];
        lb.get(ret);
        return ret;
    }

    /** {@inheritDoc} */
    @Override
    public double[] toDoubleArray() {
        if (getDataType() != DataType.FLOAT64) {
            throw new IllegalStateException(
                    "DataType mismatch, Required double" + " Actual " + getDataType());
        }
        DoubleBuffer db = toByteBuffer().asDoubleBuffer();
        double[] ret = new double[db.remaining()];
        db.get(ret);
        return ret;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mmul(NDArray other, NDArray result) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return factory.invoke("elemwise_div", new NDArray[] {this, other}, null)[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        return factory.invoke("_mul", new NDArray[] {this, other}, null)[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return factory.invoke("_sub", new NDArray[] {this, other}, null)[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mmuli(NDArray other) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mmuli(NDArray other, NDArray result) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        factory.invoke("elemwise_div", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other) {
        factory.invoke("_mul", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        factory.invoke("_sub", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amax(int... dimension) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public Number amaxNumber() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amin(int... dimension) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public Number aminNumber() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public Number max() {
        MxOpParams params = new MxOpParams();
        return factory.invoke("max", this, params).toArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return factory.invoke("max", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public Number min() {
        MxOpParams params = new MxOpParams();
        return factory.invoke("min", this, params).toArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return factory.invoke("min", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public Number sum() {
        MxOpParams params = new MxOpParams();
        return factory.invoke("sum", this, params).toArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return factory.invoke("sum", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public Number prod() {
        MxOpParams params = new MxOpParams();
        return factory.invoke("prod", this, params).toArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return factory.invoke("prod", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public Number mean() {
        MxOpParams params = new MxOpParams();
        return factory.invoke("mean", this, params).toArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("axis", axes);
        params.addParam("keepdims", keepDims);
        return factory.invoke("mean", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dup() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        return reshape(new Shape(Math.toIntExact(size())));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        if (shape.size() != -1 && getShape().size() != shape.size()) {
            throw new IllegalArgumentException("The given shape does not match the current shape");
        }
        long[] dims = Arrays.stream(shape.getShape()).asLongStream().toArray();
        Pointer pointer = JnaUtils.reshape(getHandle(), dims, false);
        return factory.create(pointer);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return factory.invoke("_npi_expand_dims", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stack(NDArray[] arrays, int axis) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        NDArray[] srcArray = new NDArray[arrays.length + 1];
        srcArray[0] = this;
        System.arraycopy(arrays, 0, srcArray, 1, arrays.length);
        return factory.invoke("_npi_stack", srcArray, params)[0];
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
        return factory.invoke("_npi_concatenate", srcArray, params)[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(double min, double max) {
        MxOpParams params = new MxOpParams();
        params.addParam("a_min", min);
        params.addParam("a_max", max);
        return factory.invoke("_npi_clip", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        MxOpParams params = new MxOpParams();
        return factory.invoke("transpose", this, params);
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
        return factory.invoke("transpose", this, params);
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
    public boolean equalsWithEps(Object o, double eps) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public boolean equalShapes(NDArray other) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int... dimension) {
        throw new UnsupportedOperationException("Not implemented yet.");
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
    public NDArray toDense() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public int nonzero() {
        MxNDArray zeros = (MxNDArray) eq(0);
        NDArray sum = factory.invoke("sum", eq(zeros).eq(zeros), null);
        return sum.toArray()[0].intValue();
    }

    /** {@inheritDoc} */
    @Override
    public boolean isEmpty() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray castTo(DataType dataType) {
        throw new UnsupportedOperationException("Not implemented yet.");
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
    public NDArray like() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ulike() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayEx getNDArrayInternal() {
        return mxNDArrayEx;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        return factory.invoke("_np_logical_not", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        return factory.invoke("_np_absolute", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return factory.invoke("_np_square", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        return factory.invoke("_np_cbrt", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        return factory.invoke("_np_floor", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        return factory.invoke("_np_ceil", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        return factory.invoke("round", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        return factory.invoke("_np_trunc", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        return factory.invoke("_npi_exp", this, null);
    }

    @Override
    public NDArray pow(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return factory.invoke("_power_scalar", this, params);
    }

    @Override
    public NDArray powi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        factory.invoke("_power_scalar", new NDArray[] {this}, new NDArray[] {this}, params);
        return this;
    }

    @Override
    public NDArray pow(NDArray other) {
        return factory.invoke("_power", new NDArray[] {this, other}, null)[0];
    }

    @Override
    public NDArray powi(NDArray other) {
        factory.invoke("_power", new NDArray[] {this, other}, new NDArray[] {this}, null);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        return factory.invoke("_np_log", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        return factory.invoke("_npi_log10", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        return factory.invoke("_np_log2", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        return factory.invoke("_npi_sin", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        return factory.invoke("_npi_cos", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        return factory.invoke("_np_tan", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        return factory.invoke("_np_arcsin", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        return factory.invoke("_np_arccos", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        return factory.invoke("_npi_arctan", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        return factory.invoke("_np_degrees", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        return factory.invoke("_np_radians", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        return factory.invoke("_npi_sinh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        return factory.invoke("_npi_cosh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return factory.invoke("_np_tanh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        return factory.invoke("_np_arcsinh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        return factory.invoke("_np_arccosh", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        return factory.invoke("_np_arctanh", this, null);
    }

    private ByteBuffer toByteBuffer() {
        Shape sh = getShape();
        DataType dType = getDataType();
        int product = sh.size();
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
            int limit = Math.min(getShape().head(), MAX_PRINT_ITEMS);
            ByteBuffer buf = toByteBuffer().slice();
            buf.limit(limit * getDataType().getNumOfBytes());
            buf.order(ByteOrder.LITTLE_ENDIAN);
            sb.append(Utils.toCharSequence(buf, getDataType()));
            int remaining = getShape().head() - limit;
            if (remaining > 0) {
                sb.append(", ... ").append(remaining).append(" more");
            }
        } else {
            sb.append(LF);
            int limit = Math.min(len, MAX_PRINT_ROWS);
            for (int i = 0; i < limit; ++i) {
                try (MxNDArray nd = (MxNDArray) get(i)) {
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
            factory = null;
        }
    }
}
