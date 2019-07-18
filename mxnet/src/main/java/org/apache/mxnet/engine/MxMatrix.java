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
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.index.NDIndex;
import software.amazon.ai.ndarray.internal.NDArrayEx;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
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
    public DataDesc getDataDescriptor() {
        return array.getDataDescriptor();
    }

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
    public Matrix asMatrix() {
        return null;
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
    public void attachGradient() {
        array.attachGradient();
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient(GradReq gradReq, SparseFormat sparseFormat) {
        array.attachGradient(gradReq, sparseFormat);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        return array.getGradient();
    }

    /** {@inheritDoc} */
    @Override
    public byte[] getEncoded() {
        return array.getEncoded();
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
    public NDArray get(NDIndex index) {
        return array.get(index);
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray arr) {
        array.copyTo(arr);
    }

    @Override
    public NDArray dup() {
        return array.dup();
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
    public boolean equalsWithEps(Object o, double eps) {
        return array.equalsWithEps(o, eps);
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
    public NDArray add(NDArray... others) {
        return array.add(others);
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
    public NDArray mul(NDArray... others) {
        return array.mul(others);
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
    public NDArray addi(Number n) {
        return array.addi(n);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray... others) {
        return array.addi(others);
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
    public NDArray muli(NDArray... others) {
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

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        return array.trace(offset, axis1, axis2);
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
    public NDArray argsort(int axis, boolean ascending) {
        return array.argsort(axis, ascending);
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
    public NDArray softmax(int[] axes) {
        return array.softmax(axes);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes, double temperature) {
        return array.softmax(axes, temperature);
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
        return array.repeat(repeats);
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
    public NDArray mmul(NDArray other) {
        return array.mmul(other);
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
    public NDArray clip(double min, double max) {
        return array.clip(min, max);
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
    public NDArray broadcast(long... shape) {
        return array.broadcast(shape);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(NDArray result) {
        return array.broadcast(result);
    }

    /** {@inheritDoc} */
    @Override
    public boolean equalShapes(NDArray other) {
        return array.equalShapes(other);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argmax() {
        return array.argmax();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argmax(int axis, boolean keepDims) {
        return array.argmax(axis, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argmin() {
        return array.argmin();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argmin(int axis, boolean keepDims) {
        return array.argmin(axis, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public Number percentileNumber(Number percentile) {
        return array.percentileNumber(percentile);
    }

    /** {@inheritDoc} */
    @Override
    public Number medianNumber() {
        return array.medianNumber();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median(int... dimension) {
        return array.median(dimension);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int... dimension) {
        return array.percentile(percentile, dimension);
    }

    /** {@inheritDoc} */
    @Override
    public long nonzero() {
        return array.nonzero();
    }

    /** {@inheritDoc} */
    @Override
    public boolean isEmpty() {
        return array.isEmpty();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        return array.logicalNot();
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
