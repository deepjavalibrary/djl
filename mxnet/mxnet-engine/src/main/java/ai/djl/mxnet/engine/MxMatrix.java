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
import ai.djl.ndarray.Matrix;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.function.Predicate;

/** {@code MxMatrix} is the MXNet implementation of {@link Matrix}. */
public class MxMatrix implements Matrix {

    private MxNDArray array;

    /**
     * Constructs the MXMatrix given a 2-D {@link MxNDArray}.
     *
     * @param array the corresponding {@link NDArray}
     */
    public MxMatrix(MxNDArray array) {
        this.array = array;
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient() {
        array.attachGradient();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        return array.getGradient();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putRow(long row, NDArray toPut) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putColumn(int column, NDArray toPut) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getScalar(long row, long column) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray diviColumnVector(NDArray columnVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divColumnVector(NDArray columnVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray diviRowVector(NDArray rowVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divRowVector(NDArray rowVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiviColumnVector(NDArray columnVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivColumnVector(NDArray columnVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiviRowVector(NDArray rowVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivRowVector(NDArray rowVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muliColumnVector(NDArray columnVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mulColumnVector(NDArray columnVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muliRowVector(NDArray rowVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mulRowVector(NDArray rowVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubiColumnVector(NDArray columnVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubColumnVector(NDArray columnVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubiRowVector(NDArray rowVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubRowVector(NDArray rowVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subiColumnVector(NDArray columnVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subColumnVector(NDArray columnVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subiRowVector(NDArray rowVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subRowVector(NDArray rowVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addiColumnVector(NDArray columnVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putiColumnVector(NDArray columnVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addColumnVector(NDArray columnVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addiRowVector(NDArray rowVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putiRowVector(NDArray rowVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addRowVector(NDArray rowVector) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getColumn(long i) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getRow(long i) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getColumns(int... columns) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getRows(int... rows) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray put(int i, int j, Number element) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        return array.reshape(shape);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(char order, int rows, int columns) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public double[][] toDoubleMatrix() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public float[][] toFloatMatrix() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public long[][] toLongMatrix() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public int[][] toIntMatrix() {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
        return array.getManager();
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return array.getName();
    }

    /** {@inheritDoc} */
    @Override
    public void setName(String name) {
        array.setName(name);
    }

    /** {@inheritDoc} */
    @Override
    public String getUid() {
        return 'M' + array.getUid();
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return array.getDataType();
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return array.getDevice();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        return array.getShape();
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        return array.getSparseFormat();
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSparse() {
        return array.isSparse();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asInDevice(Device dev, boolean copy) {
        return array.asInDevice(dev, copy);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asType(DataType dtype, boolean copy) {
        return array.asType(dtype, copy);
    }

    /** {@inheritDoc} */
    @Override
    public Matrix asMatrix() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return array.toByteBuffer();
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        array.set(data);
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, NDArray value) {
        array.set(index, value);
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, Number value) {
        array.set(index, value);
    }

    /** {@inheritDoc} */
    @Override
    public void setScalar(NDIndex index, Number value) {
        array.setScalar(index, value);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDIndex index) {
        return array.get(index);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDArray index) {
        return array.get(index);
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray arr) {
        array.copyTo(arr);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray booleanMask(NDArray index, int axis) {
        return array.booleanMask(index, axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike() {
        return array.zerosLike();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        return array.onesLike();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray like() {
        return array.like();
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(Number number) {
        return array.contentEquals(number);
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(NDArray other) {
        return array.contentEquals(other);
    }

    /** {@inheritDoc} */
    @Override
    public boolean allClose(NDArray other, double rtol, double atol, boolean equalNan) {
        return array.allClose(other, rtol, atol, equalNan);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(Number other) {
        return array.eq(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        return array.eq(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number other) {
        return array.neq(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        return array.neq(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number other) {
        return array.gt(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        return array.gt(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number other) {
        return array.gte(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        return array.gte(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number other) {
        return array.lt(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        return array.lt(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number other) {
        return array.lte(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        return array.lte(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        return array.add(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        return array.add(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        return array.sub(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return array.sub(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        return array.mul(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        return array.mul(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        return array.div(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return array.div(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        return array.mod(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        return array.mod(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(Number n) {
        return array.pow(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(NDArray other) {
        return array.pow(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(Number n) {
        return array.maximum(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(NDArray other) {
        return array.maximum(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(Number n) {
        return array.maximum(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(NDArray other) {
        return array.maximum(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        return array.addi(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        return array.addi(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        return array.subi(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        return array.subi(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        return array.muli(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray others) {
        return array.muli(others);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        return array.divi(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        return array.divi(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        return array.modi(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        return array.modi(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(Number n) {
        return array.powi(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(NDArray other) {
        return array.powi(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        return array.neg();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        return array.negi();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        return array.abs();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return array.square();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        return array.cbrt();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        return array.floor();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        return array.ceil();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        return array.round();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        return array.trunc();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        return array.exp();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        return array.log();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        return array.log10();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        return array.log2();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        return array.sin();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        return array.cos();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        return array.tan();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        return array.asin();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        return array.acos();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        return array.atan();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        return array.sinh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toSparse(SparseFormat fmt) {
        return array.toSparse(fmt);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDense() {
        return array.toDense();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        return array.cosh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return array.tanh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        return array.asinh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        return array.acosh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        return array.atanh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        return array.toDegrees();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        return array.toRadians();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max() {
        return array.max();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes) {
        return array.max(axes);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        return array.max(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min() {
        return array.min();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes) {
        return array.min(axes);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        return array.min(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum() {
        return array.sum();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        return array.sum(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod() {
        return array.prod();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        return array.prod(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        return array.mean();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        return array.mean(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        return array.trace(offset, axis1, axis2);
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int[] indices, int axis) {
        return array.split(indices, axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        return array.flatten();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        return array.expandDims(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        return array.squeeze(axes);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalAnd(NDArray other) {
        return array.logicalAnd(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray other) {
        return array.logicalOr(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray other) {
        return array.logicalXor(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        return array.logicalNot();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argSort(int axis, boolean ascending) {
        return array.argSort(axis, ascending);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        return array.sort(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        return array.sort();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes, double temperature) {
        return array.softmax(axes, temperature);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum(int axis) {
        return array.cumSum(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum() {
        return array.cumSum();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        return array.isInfinite();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        return array.isNaN();
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
        return array.tile(repeats);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, long repeats) {
        return array.tile(axis, repeats);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long[] repeats) {
        return array.tile(repeats);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(Shape desiredShape) {
        return array.tile(desiredShape);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long repeats) {
        return array.repeat(repeats);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int axis, long repeats) {
        return array.repeat(axis, repeats);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long[] repeats) {
        return array.repeat(repeats);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(Shape desiredShape) {
        return array.repeat(desiredShape);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dot(NDArray other) {
        return array.dot(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(Number min, Number max) {
        return array.clip(min, max);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray swapAxes(int axis1, int axis2) {
        return array.swapAxes(axis1, axis2);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        return array.transpose();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int... dimensions) {
        return array.transpose(dimensions);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        return array.broadcast(shape);
    }

    /** {@inheritDoc} */
    @Override
    public boolean shapeEquals(NDArray other) {
        return array.shapeEquals(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax() {
        return array.argMax();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        return array.argMax(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin() {
        return array.argMin();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin(int axis) {
        return array.argMin(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile) {
        return array.percentile(percentile);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int[] dimension) {
        return array.percentile(percentile, dimension);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median() {
        return array.median();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median(int[] axes) {
        return array.median(axes);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray nonzero() {
        return array.nonzero();
    }

    /** {@inheritDoc} */
    @Override
    public boolean isEmpty() {
        return array.isEmpty();
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayEx getNDArrayInternal() {
        return array.getNDArrayInternal();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        // DO NOTHING
    }
}
