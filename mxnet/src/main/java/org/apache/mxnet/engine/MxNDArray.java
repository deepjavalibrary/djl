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

import com.amazon.ai.Context;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Layout;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;
import com.amazon.ai.util.PairList;
import com.amazon.ai.util.Utils;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import java.io.OutputStream;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.Condition;
import org.apache.mxnet.jna.FunctionInfo;
import org.apache.mxnet.jna.JnaUtils;

public class MxNDArray extends NativeResource implements NDArray {

    private static final Map<String, FunctionInfo> OPS = JnaUtils.getNdArrayFunctions();

    private static final int MAX_DEPTH = 10;
    private static final int MAX_PRINT_ROWS = 10;
    private static final int MAX_PRINT_ITEMS = 20;
    private static final String LF = System.getProperty("line.separator");

    private Context context;
    private SparseFormat sparseFormat;
    private DataType dataType;
    private Shape shape;
    private MxNDFactory factory;
    private boolean isReady;

    MxNDArray(
            MxNDFactory factory,
            Context context,
            SparseFormat sparseFormat,
            Shape shape,
            DataType dataType,
            Pointer handle) {
        super(handle);
        this.factory = factory;
        this.context = context;
        this.dataType = dataType;
        this.shape = shape;
        this.sparseFormat = sparseFormat;
        this.isReady = false;
    }

    MxNDArray(
            MxNDFactory factory,
            Context context,
            SparseFormat sparseFormat,
            Shape shape,
            DataType dataType) {
        this(
                factory,
                context,
                sparseFormat,
                shape,
                dataType,
                JnaUtils.createNdArray(
                        context,
                        shape,
                        dataType,
                        shape.dimension(),
                        sparseFormat != SparseFormat.DEFAULT));
    }

    /** {@inheritDoc} */
    @Override
    public byte[] getEncoded() {
        return new byte[0];
    }

    /** {@inheritDoc} */
    @Override
    public void encode(OutputStream os) {}

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
        return new DataDesc(
                getShape(), getDataType(), null, getLayout(), getContext(), getSparseFormat());
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        if (data.remaining() != getShape().size()) {
            throw new IllegalArgumentException(
                    "array size ("
                            + data.remaining()
                            + ")do not match the size of NDArray ("
                            + getShape().size());
        }
        JnaUtils.syncCopyFromCPU(getHandle(), data);
    }

    /** {@inheritDoc} */
    @Override
    public void set(List<Float> data) {
        waitToWrite();
        int size = data.size();
        FloatBuffer output =
                ByteBuffer.allocateDirect(size * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        for (Float v : data) {
            output.put(v);
        }
        output.rewind();
        set(output);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray at(int index) {
        Pointer pointer = JnaUtils.ndArrayAt(getHandle(), index);
        return factory.create(pointer);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray slice(int begin, int end) {
        Pointer pointer = JnaUtils.slice(getHandle(), begin, end);
        return factory.create(pointer);
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

        FunctionInfo functionInfo = OPS.get("_copyto");

        MxNDArray array = (MxNDArray) ndArray;

        functionInfo.invoke(factory, this, new NDList(array), null);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray asInContext(Context ctx, boolean copy) {
        if (ctx.equals(getContext()) && !copy) {
            return this;
        }
        MxNDArray nd = factory.create(ctx, getShape(), getDataType(), getSparseFormat());
        copyTo(nd);
        return nd;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray asType(DataType dtype, boolean copy) {
        if (dtype.equals(getDataType()) && !copy) {
            return this;
        }
        MxNDArray nd = factory.create(getContext(), getShape(), dtype, getSparseFormat());
        copyTo(nd);
        return nd;
    }

    /** {@inheritDoc} */
    public void waitToRead() {
        if (!isReady) {
            JnaUtils.waitToRead(getHandle());
            isReady = true;
        }
    }

    public void waitToWrite() {
        if (!isReady) {
            JnaUtils.waitToWrite(getHandle());
            isReady = true;
        }
    }

    public void waitAll() {
        JnaUtils.waitToRead(getHandle());
        isReady = true;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argsort(int axis, boolean isAscend) {
        PairList<String, String> params = new PairList<>();
        params.add("axis", String.valueOf(axis));
        params.add("is_ascend", isAscend ? "True" : "False");

        FunctionInfo functionInfo = OPS.get("argsort");
        return functionInfo.invoke(factory, this, params).get(0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(Integer axis, Double temperature) {
        PairList<String, String> params = new PairList<>();
        if (axis != null) {
            params.add("axis", String.valueOf(axis));
        }
        if (temperature != null) {
            params.add("temperature", String.valueOf(temperature));
        }

        FunctionInfo functionInfo = OPS.get("softmax");
        return functionInfo.invoke(factory, this, params).get(0);
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int numOutputs, Integer axis, Boolean squeezeAxis) {
        PairList<String, String> params = new PairList<>();
        params.add("num_outputs", String.valueOf(numOutputs));
        if (axis != null) {
            params.add("axis", String.valueOf(axis));
        }
        if (squeezeAxis != null) {
            params.add("squeeze_axis", String.valueOf(squeezeAxis));
        }

        FunctionInfo functionInfo = OPS.get("split");
        return functionInfo.invoke(factory, this, params);
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSparse() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsumi(int dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsum(int dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray assign(NDArray arr) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray assignIf(NDArray arr, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray replaceWhere(NDArray arr, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putScalar(long value, long... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putScalar(double value, long... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putScalar(float value, long... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putScalar(int value, long... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eps(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eps(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray match(NDArray comp, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray match(Number comp, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getWhere(NDArray comp, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getWhere(Number comp, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putWhere(NDArray comp, NDArray put, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putWhere(Number comp, NDArray put, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putWhereWithMask(NDArray mask, NDArray put) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putWhereWithMask(NDArray mask, Number put) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putWhere(Number comp, Number put, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDArray indices) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(List<List<Integer>> indices) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getColumns(int... columns) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getRows(int... rows) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray assign(Number value) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putSlice(int slice, NDArray put) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cond(Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repmat(int... shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int dimension, long... repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putRow(long row, NDArray toPut) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putColumn(int column, NDArray toPut) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getScalar(long row, long column) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getScalar(long i) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public double squaredDistance(NDArray other) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public double distance2(NDArray other) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public double distance1(NDArray other) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray put(List<List<Integer>> indices, NDArray element) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray put(NDArray indices, NDArray element) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray put(NDArray element, int... indices) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray put(int i, int j, Number element) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray put(int i, NDArray element) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray diviColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray diviRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiviColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiviRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muliColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mulColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muliRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mulRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubiColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubiRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subiColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subiRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addiColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putiColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addiRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putiRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mmul(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public double[][] toDoubleMatrix() {
        return new double[0][];
    }

    /** {@inheritDoc} */
    @Override
    public double[] toDoubleArray() {
        return new double[0];
    }

    public FunctionInfo genericNDArrayFunctionInvoke(String opName, Map<String, Object> args) {
        FunctionInfo func = OPS.get(opName);
        if (func == null) {
            throw new UnsupportedOperationException("Unsupported operation: " + opName);
        }

        return func;
    }

    /** {@inheritDoc} */
    @Override
    public float[] toFloatArray() {
        FloatBuffer fb = toByteBuffer().asFloatBuffer();
        float[] ret = new float[fb.remaining()];
        fb.get(ret);
        return ret;
    }

    /** {@inheritDoc} */
    @Override
    public float[][] toFloatMatrix() {
        return new float[0][];
    }

    /** {@inheritDoc} */
    @Override
    public int[] toIntArray() {
        return new int[0];
    }

    /** {@inheritDoc} */
    @Override
    public long[] toLongArray() {
        return new long[0];
    }

    /** {@inheritDoc} */
    @Override
    public long[][] toLongMatrix() {
        return new long[0][];
    }

    /** {@inheritDoc} */
    @Override
    public int[][] toIntMatrix() {
        return new int[0][];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mmul(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mmuli(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mmuli(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray normmax(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number normmaxNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm2(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number norm2Number() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm1(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number norm1Number() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray std(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number stdNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray std(boolean biasCorrected, int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number stdNumber(boolean biasCorrected) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(NDArray result, int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amean(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number meanNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number ameanNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray var(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray var(boolean biasCorrected, int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number varNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amax(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number maxNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number amaxNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amin(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number minNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number aminNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(boolean keepDims, int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(NDArray result, int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number sumNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number entropyNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number shannonEntropyNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number logEntropyNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray entropy(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray shannonEntropy(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logEntropy(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getScalar(int... indices) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getScalar(long... indices) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public long getLong(int... indices) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public long getLong(long... indices) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public double getDouble(int... indices) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public double getDouble(long... indices) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public float getFloat(int... indices) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public float getFloat(long... indices) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dup() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ravel() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ravel(char order) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray slice(long i, int dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray slice(long i) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(char order, long... newShape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(char order, int... newShape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(char order, int rows, int columns) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(long... newShape) {
        Pointer pointer = JnaUtils.reshape(getHandle(), newShape, false);
        return factory.create(pointer);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(int[] shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transposei() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray swapAxes(int dimension, int with) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int... dimensions) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transposei(int... dimensions) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getColumn(long i) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getRow(long i) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public long size(int dimension) {
        return shape.size(dimension);
    }

    @Override
    public long size() {
        return shape.size();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(long... shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Object element() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equalsWithEps(Object o, double eps) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equalShapes(NDArray other) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray remainder(NDArray denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray remainder(NDArray denominator, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray remainder(Number denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray remainder(Number denominator, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray remainderi(NDArray denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray remainderi(Number denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fmod(NDArray denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fmod(NDArray denominator, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fmod(Number denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fmod(Number denominator, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fmodi(NDArray denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fmodi(Number denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number percentileNumber(Number percentile) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number medianNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDense() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public int nonzero() {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isEmpty() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray castTo(DataType dataType) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean all() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public boolean any() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public boolean none() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray like() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ulike() {
        return null;
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
                try (MxNDArray nd = at(i)) {
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
