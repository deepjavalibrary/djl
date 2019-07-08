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

import java.nio.Buffer;
import java.util.function.Predicate;
import software.amazon.ai.Context;
import software.amazon.ai.ndarray.Matrix;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDFactory;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.index.NDIndex;
import software.amazon.ai.ndarray.internal.NDArrayEx;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Layout;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.ndarray.types.SparseFormat;
import software.amazon.ai.training.GradReq;

public class MxMatrix implements Matrix {

    private MxNDArray array;

    public MxMatrix(MxNDArray array) {
        this.array = array;
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
    public NDArray reshape(char order, int rows, int columns) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        return array.transpose();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int[] dimensions) {
        return array.transpose(dimensions);
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
    public byte[] getEncoded() {
        return array.getEncoded();
    }

    /** {@inheritDoc} */
    @Override
    public NDFactory getFactory() {
        return array.getFactory();
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return array.getDataType();
    }

    /** {@inheritDoc} */
    @Override
    public Context getContext() {
        return array.getContext();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        return array.getShape();
    }

    /** {@inheritDoc} */
    @Override
    public Layout getLayout() {
        return array.getLayout();
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc getDataDescriptor() {
        return array.getDataDescriptor();
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        array.set(data);
    }

    /** {@inheritDoc} */
    @Override
    public void set(float[] data) {
        array.set(data);
    }

    /** {@inheritDoc} */
    @Override
    public void set(int[] data) {
        array.set(data);
    }

    /** {@inheritDoc} */
    @Override
    public void set(double[] data) {
        array.set(data);
    }

    /** {@inheritDoc} */
    @Override
    public void set(long[] data) {
        array.set(data);
    }

    /** {@inheritDoc} */
    @Override
    public void set(byte[] data) {
        array.set(data);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDIndex index) {
        return array.get(index);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray set(NDIndex index, NDArray value) {
        return array.set(index, value);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray set(NDIndex index, Number value) {
        return array.set(index, value);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray setElement(NDIndex index, Number value) throws IllegalArgumentException {
        return array.setElement(index, value);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray seti(NDIndex index, NDArray value) {
        return array.seti(index, value);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray seti(NDIndex index, Number value) {
        return array.seti(index, value);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray setElementi(NDIndex index, Number value) throws IllegalArgumentException {
        return array.setElementi(index, value);
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray arr) {
        array.copyTo(arr);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asInContext(Context ctx, boolean copy) {
        return array.asInContext(ctx, copy);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asType(DataType dtype, boolean copy) {
        return array.asType(dtype, copy);
    }

    /** {@inheritDoc} */
    @Override
    public void attachGrad() {
        array.attachGrad();
    }

    /** {@inheritDoc} */
    @Override
    public void attachGrad(GradReq gradReq, SparseFormat sparseFormat) {
        array.attachGrad(gradReq, sparseFormat);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        return array.getGradient();
    }

    /** {@inheritDoc} */
    @Override
    public void backward() {
        array.backward();
    }

    /** {@inheritDoc} */
    @Override
    public void backward(boolean retainGraph, boolean isTraining) {
        array.backward(retainGraph, isTraining);
    }

    /** {@inheritDoc} */
    @Override
    public void backward(NDArray outGrad, boolean retainGraph, boolean isTraining) {
        array.backward(outGrad, retainGraph, isTraining);
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
    public NDArray argsort(int axis, boolean ascending) {
        return array.argsort(axis, ascending);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes, double temperature) {
        return array.softmax(axes, temperature);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes) {
        return array.softmax(axes);
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int axis, boolean squeezeAxis) {
        return array.split(axis, squeezeAxis);
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int axis, int numOutputs) {
        return array.split(axis, numOutputs);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        return array.add(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        return array.addi(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        return array.add(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        return array.addi(other);
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
    public boolean isSparse() {
        return array.isSparse();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsumi(int axis) {
        return array.cumsumi(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsumi() {
        return array.cumsumi();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsum(int axis) {
        return array.cumsum(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsum() {
        return array.cumsum();
    }

    @Override
    public NDArray eps(Number other) {
        return array.eps(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eps(NDArray other) {
        return array.eps(other);
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
    public boolean contentEquals(NDArray other) {
        return array.contentEquals(other);
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(Number number) {
        return array.contentEquals(number);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number other) {
        return array.gt(other);
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
    public NDArray gt(NDArray other) {
        return array.gt(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number other) {
        return array.gte(other);
    }

    @Override
    public NDArray gte(NDArray other) {
        return array.gte(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number other) {
        return array.lte(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number other) {
        return array.lt(other);
    }

    @Override
    public NDArray lte(NDArray other) {
        return array.lte(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        return array.lt(other);
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
    public NDArray div(Number n) {
        return array.div(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        return array.divi(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        return array.mod(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        return array.modi(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        return array.mul(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        return array.muli(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        return array.sub(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        return array.subi(n);
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
        return array.repeat(repeats);
    }

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

    @Override
    public NDArray repeat(long[] repeats) {
        return array.repeat(repeats);
    }

    @Override
    public NDArray repeat(Shape desiredShape) {
        return array.repeat(desiredShape);
    }

    @Override
    public NDArray mmul(NDArray other) {
        return array.mmul(other);
    }

    /** {@inheritDoc} */
    @Override
    public double[] toDoubleArray() {
        return array.toDoubleArray();
    }

    /** {@inheritDoc} */
    @Override
    public float[] toFloatArray() {
        return array.toFloatArray();
    }

    /** {@inheritDoc} */
    @Override
    public int[] toIntArray() {
        return array.toIntArray();
    }

    /** {@inheritDoc} */
    @Override
    public long[] toLongArray() {
        return array.toLongArray();
    }

    /** {@inheritDoc} */
    @Override
    public byte[] toByteArray() {
        return array.toByteArray();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return array.div(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        return array.mod(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        return array.mul(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return array.sub(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        return array.divi(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        return array.modi(other);
    }

    @Override
    public NDArray argMax(int axis, boolean keepDims) {
        return null;
    }

    @Override
    public NDArray argMin() {
        return null;
    }

    @Override
    public NDArray argMin(int axis, boolean keepDims) {
        return null;
    }

    @Override
    public NDArray argMax() {
        return null;
    }

    @Override
    public Number percentileNumber(Number percentile) {
        return null;
    }

    @Override
    public Number medianNumber() {
        return null;
    }

    @Override
    public NDArray median(int... dimension) {
        return null;
    }

    @Override
    public NDArray percentile(Number percentile, int... dimension) {
        return null;
    }

    @Override
    public NDArray toDense() {
        return null;
    }

    @Override
    public long nonzero() {
        return 0;
    }

    @Override
    public boolean isEmpty() {
        return false;
    }

    @Override
    public Matrix asMatrix() {
        return null;
    }

    @Override
    public NDArray like() {
        return null;
    }

    @Override
    public NDArrayEx getNDArrayInternal() {
        return null;
    }

    @Override
    public NDArray logicalNot() {
        return null;
    }

    @Override
    public NDArray abs() {
        return null;
    }

    @Override
    public NDArray square() {
        return null;
    }

    @Override
    public NDArray cbrt() {
        return null;
    }

    @Override
    public NDArray floor() {
        return null;
    }

    @Override
    public NDArray ceil() {
        return null;
    }

    @Override
    public NDArray round() {
        return null;
    }

    @Override
    public NDArray trunc() {
        return null;
    }

    @Override
    public NDArray exp() {
        return null;
    }

    @Override
    public NDArray pow(Number n) {
        return null;
    }

    @Override
    public NDArray powi(Number n) {
        return null;
    }

    @Override
    public NDArray pow(NDArray other) {
        return null;
    }

    @Override
    public NDArray powi(NDArray other) {
        return null;
    }

    @Override
    public NDArray log() {
        return null;
    }

    @Override
    public NDArray log10() {
        return null;
    }

    @Override
    public NDArray log2() {
        return null;
    }

    @Override
    public NDArray sin() {
        return null;
    }

    @Override
    public NDArray cos() {
        return null;
    }

    @Override
    public NDArray tan() {
        return null;
    }

    @Override
    public NDArray asin() {
        return null;
    }

    @Override
    public NDArray acos() {
        return null;
    }

    @Override
    public NDArray atan() {
        return null;
    }

    @Override
    public NDArray toDegrees() {
        return null;
    }

    @Override
    public NDArray toRadians() {
        return null;
    }

    @Override
    public NDArray sinh() {
        return null;
    }

    @Override
    public NDArray cosh() {
        return null;
    }

    @Override
    public NDArray tanh() {
        return null;
    }

    @Override
    public NDArray asinh() {
        return null;
    }

    @Override
    public NDArray acosh() {
        return null;
    }

    @Override
    public NDArray atanh() {
        return null;
    }

    @Override
    public void close() {}

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other) {
        return array.muli(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        return array.subi(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amax(int... dimension) {
        return array.amax(dimension);
    }

    /** {@inheritDoc} */
    @Override
    public Number amaxNumber() {
        return array.amaxNumber();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amin(int... dimension) {
        return array.amin(dimension);
    }

    /** {@inheritDoc} */
    @Override
    public Number aminNumber() {
        return array.aminNumber();
    }

    /** {@inheritDoc} */
    @Override
    public Number max() {
        return array.max();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        return array.max(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public Number min() {
        return array.min();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        return array.min(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public Number sum() {
        return array.sum();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        return array.sum(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public Number prod() {
        return array.prod();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        return array.prod(axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public Number mean() {
        return array.mean();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        return array.mean(axes, keepDims);
    }

    @Override
    public NDArray dup() {
        return array.dup();
    }

    @Override
    public NDArray flatten() {
        return array.flatten();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        return array.reshape(shape);
    }

    @Override
    public NDArray expandDims(int axis) {
        return array.expandDims(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stack(NDArray[] arrays, int axis) {
        return array.stack(arrays, axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stack(NDList arrays, int axis) {
        return array.stack(arrays, axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray concat(NDArray[] arrays, int axis) {
        return array.concat(arrays, axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(double min, double max) {
        return array.clip(min, max);
    }

    /** {@inheritDoc} */
    @Override
    public long size(int dimension) {
        return array.size(dimension);
    }

    /** {@inheritDoc} */
    @Override
    public long size() {
        return array.size();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(long... shape) {
        return array.broadcast(shape);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(NDArray result) {
        return array.broadcast(result);
    }

    @Override
    public boolean equalsWithEps(Object o, double eps) {
        return array.equalsWithEps(o, eps);
    }

    @Override
    public boolean equalShapes(NDArray other) {
        return false;
    }
}
