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
package org.tensorflow.engine;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;
import software.amazon.ai.Context;
import software.amazon.ai.ndarray.Matrix;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.index.NDIndex;
import software.amazon.ai.ndarray.internal.NDArrayEx;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.ndarray.types.SparseFormat;
import software.amazon.ai.training.GradReq;

public class TfNDArray implements NDArray {

    private Tensor<?> tensor;
    private Output<?> out;
    private Shape shape;
    private TfNDManager manager;

    TfNDArray(NDManager manager, Tensor<?> tensor) {
        this.manager = (TfNDManager) manager;
        this.manager.attach(this);
        this.tensor = tensor;
    }

    TfNDArray(NDManager manager, Output<?> out) {
        this.manager = (TfNDManager) manager;
        this.manager.attach(this);
        this.out = out;
    }

    public TfNDArray(NDManager manager, Shape shape, FloatBuffer data) {
        this.manager = (TfNDManager) manager;
        this.manager.attach(this);
        tensor = Tensor.create(shape.getShape(), data);
        this.shape = shape;
    }

    TfNDArray(NDManager manager, Shape shape, ByteBuffer data) {
        this.manager = (TfNDManager) manager;
        tensor = Tensor.create(UInt8.class, shape.getShape(), data);
        this.shape = shape;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return TfDataType.fromTf(getTfDataType());
    }

    /** {@inheritDoc} */
    @Override
    public Context getContext() {
        return manager.getContext();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape == null) {
            runToTensor();
            shape = new Shape(tensor.shape());
        }
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc getDataDescriptor() {
        return new DataDesc(getShape(), getDataType(), null);
    }

    private void runToTensor() {
        if (tensor == null) {
            tensor = manager.getSession().runner().fetch(out.op().name()).run().get(0);
        }
    }

    public org.tensorflow.DataType getTfDataType() {
        if (tensor != null) {
            return tensor.dataType();
        } else {
            return out.dataType();
        }
    }

    @Override
    public SparseFormat getSparseFormat() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSparse() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asInContext(Context ctx, boolean copy) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asType(DataType dtype, boolean copy) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Matrix asMatrix() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void backward() {}

    /** {@inheritDoc} */
    @Override
    public void backward(boolean retainGraph, boolean isTraining) {}

    /** {@inheritDoc} */
    @Override
    public void backward(NDArray outGrad, boolean retainGraph, boolean isTraining) {}

    /** {@inheritDoc} */
    @Override
    public void attachGradient() {}

    /** {@inheritDoc} */
    @Override
    public void attachGradient(GradReq gradReq, SparseFormat sparseFormat) {}

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public byte[] getEncoded() {
        return new byte[0];
    }

    /** {@inheritDoc} */
    @Override
    public double[] toDoubleArray() {
        if (getDataType() != DataType.FLOAT64) {
            throw new IllegalStateException(
                    "DataType mismatch, Required double" + " Actual " + getDataType());
        }
        runToTensor();
        DoubleBuffer db = DoubleBuffer.allocate(Math.toIntExact(size()));
        tensor.writeTo(db.duplicate());
        double[] ret = new double[Math.toIntExact(size())];
        db.get(ret);
        return ret;
    }

    /** {@inheritDoc} */
    @Override
    public float[] toFloatArray() {
        if (getDataType() != DataType.FLOAT32) {
            throw new IllegalStateException(
                    "DataType mismatch, Required float" + " Actual " + getDataType());
        }
        runToTensor();
        FloatBuffer fb = FloatBuffer.allocate(Math.toIntExact(size()));
        tensor.writeTo(fb.duplicate());
        float[] ret = new float[Math.toIntExact(size())];
        fb.get(ret);
        return ret;
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
    public byte[] toByteArray() {
        return new byte[0];
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }

    /** {@inheritDoc} */
    @Override
    public void set(float[] data) {}

    /** {@inheritDoc} */
    @Override
    public void set(int[] data) {}

    /** {@inheritDoc} */
    @Override
    public void set(double[] data) {}

    /** {@inheritDoc} */
    @Override
    public void set(long[] data) {}

    /** {@inheritDoc} */
    @Override
    public void set(byte[] data) {}

    /** {@inheritDoc} */
    @Override
    public NDArray set(NDIndex index, NDArray value) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray set(NDIndex index, Number value) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray setElement(NDIndex index, Number value) throws IllegalArgumentException {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray seti(NDIndex index, NDArray value) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray seti(NDIndex index, Number value) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray setElementi(NDIndex index, Number value) throws IllegalArgumentException {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDIndex index) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray array) {}

    /** {@inheritDoc} */
    @Override
    public NDArray dup() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike() {
        Operation op =
                manager.getGraph()
                        .opBuilder("ZerosLike", "ZerosLike_" + TfNDManager.nextNameAssignment())
                        .setAttr("T", getTfDataType())
                        .addInput(getOutput())
                        .build();
        return new TfNDArray(manager, op.output(0));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        Operation op =
                manager.getGraph()
                        .opBuilder("OnesLike", "OnesLike_" + TfNDManager.nextNameAssignment())
                        .setAttr("T", getTfDataType())
                        .addInput(getOutput())
                        .build();
        return new TfNDArray(manager, op.output(0));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray like() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(Number number) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(NDArray other) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equalsWithEps(Object o, double eps) {
        return false;
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
    public NDArray gt(Number other) {
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
    public NDArray gte(NDArray other) {
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
    public NDArray lte(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray... others) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray... others) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray... others) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray... others) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        return null;
    }

    @Override
    public NDArray toSparse(SparseFormat fmt) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(NDArray other) {
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
    public NDArray abs() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number max() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number min() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number sum() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number prod() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number mean() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int axis, boolean squeezeAxis) {
        TfNDArray axisOp = (TfNDArray) manager.create(axis);
        Operation op =
                manager.getGraph()
                        .opBuilder("Split", "Split_" + TfNDManager.nextNameAssignment())
                        .setAttr("T", getTfDataType())
                        .setAttr("num_split", size(axis))
                        .addInput(axisOp.getOutput())
                        .addInput(getOutput())
                        .build();

        NDArray[] result =
                IntStream.range(0, op.numOutputs())
                        .mapToObj((int i) -> new TfNDArray(manager, op.output(i)))
                        .toArray(NDArray[]::new);
        return new NDList(result);
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int axis, int numOutputs) throws IllegalArgumentException {
        if (axis < 0 || axis > getShape().dimension()) {
            throw new IllegalArgumentException("Invalid axis value");
        }
        if (numOutputs < 0 || numOutputs > size(axis)) {
            throw new IllegalArgumentException("Invalid numOutputs");
        }
        TfNDArray axisOp = (TfNDArray) manager.create(axis);
        Operation op =
                manager.getGraph()
                        .opBuilder("Split", "Split_" + TfNDManager.nextNameAssignment())
                        .setAttr("T", getTfDataType())
                        .setAttr("num_split", numOutputs)
                        .addInput(axisOp.getOutput())
                        .addInput(getOutput())
                        .build();

        NDArray[] result =
                IntStream.range(0, op.numOutputs())
                        .mapToObj((int i) -> new TfNDArray(manager, op.output(i)))
                        .toArray(NDArray[]::new);
        return new NDList(result);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stack(NDArray[] arrays, int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stack(NDList arrays, int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray concat(NDArray[] arrays, int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argsort(int axis, boolean ascending) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes, double temperature) {
        Operation op =
                manager.getGraph()
                        .opBuilder("Softmax", "Softmax_" + TfNDManager.nextNameAssignment())
                        .setAttr("T", getTfDataType())
                        .addInput(getOutput())
                        .build();

        return new TfNDArray(manager, op.output(0));
    }

    Output<?> getOutput() {
        if (out == null) {
            Operation op =
                    manager.getGraph()
                            .opBuilder("Const", "Const_" + TfNDManager.nextNameAssignment())
                            .setAttr("dtype", tensor.dataType())
                            .setAttr("value", tensor)
                            .build();
            out = op.output(0);
        }
        return out;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsumi(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsumi() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsum(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsum() {
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
    public NDArray createMask(NDIndex index) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createMask(Predicate<Number> predicate) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, long repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long[] repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(Shape desiredShape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int axis, long repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long[] repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(Shape desiredShape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mmul(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amax(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number amaxNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amin(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number aminNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(double min, double max) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int[] dimensions) {
        return null;
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
    public boolean equalShapes(NDArray other) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argmax() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argmax(int axis, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argmin() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argmin(int axis, boolean keepDims) {
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
    public long nonzero() {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isEmpty() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public boolean none() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayEx getNDArrayInternal() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (tensor != null) {
            tensor.close();
        }
        tensor = null;
    }

    public Tensor<?> getTensor() {
        return tensor;
    }
}
