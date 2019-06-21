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
package software.amazon.ai.test.mock;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.Buffer;
import java.util.List;
import java.util.concurrent.locks.Condition;
import software.amazon.ai.Context;
import software.amazon.ai.ndarray.Matrix;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDFactory;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.internal.NDArrayEx;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Layout;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.ndarray.types.SparseFormat;
import software.amazon.ai.training.GradReq;

public class MockNDArray implements NDArray {

    @Override
    public byte[] getEncoded() {
        return new byte[0];
    }

    @Override
    public void encode(OutputStream os) throws IOException {}

    @Override
    public NDFactory getFactory() {
        return null;
    }

    @Override
    public DataType getDataType() {
        return null;
    }

    @Override
    public Context getContext() {
        return null;
    }

    @Override
    public Shape getShape() {
        return null;
    }

    @Override
    public Layout getLayout() {
        return null;
    }

    @Override
    public DataDesc getDataDescriptor() {
        return null;
    }

    @Override
    public void set(Buffer data) {}

    @Override
    public void set(float[] data) {}

    @Override
    public void set(int[] data) {}

    @Override
    public void set(double[] data) {}

    @Override
    public void set(long[] data) {}

    @Override
    public void set(byte[] data) {}

    @Override
    public NDArray at(int index) {
        return null;
    }

    @Override
    public NDArray slice(int begin, int end) {
        return null;
    }

    @Override
    public void copyTo(NDArray array) {}

    @Override
    public NDArray asInContext(Context ctx, boolean copy) {
        return null;
    }

    @Override
    public NDArray asType(DataType dtype, boolean copy) {
        return null;
    }

    @Override
    public void attachGrad() {}

    @Override
    public void attachGrad(GradReq gradReq, SparseFormat sparseFormat) {}

    @Override
    public NDArray getGradient() {
        return null;
    }

    @Override
    public void backward() {}

    @Override
    public void backward(boolean retainGraph, boolean isTraining) {}

    @Override
    public void backward(NDArray outGrad, boolean retainGraph, boolean isTraining) {}

    @Override
    public NDArray argsort(int axis, boolean ascending) {
        return null;
    }

    @Override
    public NDArray softmax(int[] axes) {
        return null;
    }

    @Override
    public NDArray softmax(int[] axes, double temperature) {
        return null;
    }

    @Override
    public NDList split(int axis, boolean squeezeAxis) {
        return null;
    }

    @Override
    public NDList split(int axis, int numOutputs) {
        return null;
    }

    @Override
    public NDArray zerosLike() {
        return null;
    }

    @Override
    public NDArray onesLike() {
        return null;
    }

    @Override
    public boolean isSparse() {
        return false;
    }

    @Override
    public NDArray cumsumi(int dimension) {
        return null;
    }

    @Override
    public NDArray cumsum(int dimension) {
        return null;
    }

    @Override
    public NDArray assign(NDArray arr) {
        return null;
    }

    @Override
    public NDArray assignIf(NDArray arr, Condition condition) {
        return null;
    }

    @Override
    public NDArray replaceWhere(NDArray arr, Condition condition) {
        return null;
    }

    @Override
    public NDArray putScalar(long value, long... dimension) {
        return null;
    }

    @Override
    public NDArray putScalar(double value, long... dimension) {
        return null;
    }

    @Override
    public NDArray putScalar(float value, long... dimension) {
        return null;
    }

    @Override
    public NDArray putScalar(int value, long... dimension) {
        return null;
    }

    @Override
    public NDArray eps(Number other) {
        return null;
    }

    @Override
    public NDArray eps(NDArray other) {
        return null;
    }

    @Override
    public NDArray eq(Number other) {
        return null;
    }

    @Override
    public NDArray eq(NDArray other) {
        return null;
    }

    @Override
    public boolean contentEquals(NDArray other) {
        return false;
    }

    @Override
    public boolean contentEquals(Number number) {
        return false;
    }

    @Override
    public NDArray gt(Number other) {
        return null;
    }

    @Override
    public NDArray neq(Number other) {
        return null;
    }

    @Override
    public NDArray neq(NDArray other) {
        return null;
    }

    @Override
    public NDArray gt(NDArray other) {
        return null;
    }

    @Override
    public NDArray gte(Number other) {
        return null;
    }

    @Override
    public NDArray gte(NDArray other) {
        return null;
    }

    @Override
    public NDArray lte(Number other) {
        return null;
    }

    @Override
    public NDArray lt(Number other) {
        return null;
    }

    @Override
    public NDArray lte(NDArray other) {
        return null;
    }

    @Override
    public NDArray lt(NDArray other) {
        return null;
    }

    @Override
    public NDArray isInfinite() {
        return null;
    }

    @Override
    public NDArray isNaN() {
        return null;
    }

    @Override
    public NDArray neg() {
        return null;
    }

    @Override
    public NDArray negi() {
        return null;
    }

    @Override
    public NDArray rdiv(Number n) {
        return null;
    }

    @Override
    public NDArray rdivi(Number n) {
        return null;
    }

    @Override
    public NDArray rsub(Number n) {
        return null;
    }

    @Override
    public NDArray rsubi(Number n) {
        return null;
    }

    @Override
    public NDArray div(Number n) {
        return null;
    }

    @Override
    public NDArray divi(Number n) {
        return null;
    }

    @Override
    public NDArray mul(Number n) {
        return null;
    }

    @Override
    public NDArray muli(Number n) {
        return null;
    }

    @Override
    public NDArray sub(Number n) {
        return null;
    }

    @Override
    public NDArray subi(Number n) {
        return null;
    }

    @Override
    public NDArray add(Number n) {
        return null;
    }

    @Override
    public NDArray addi(Number n) {
        return null;
    }

    @Override
    public NDArray addi(NDArray other) {
        return null;
    }

    @Override
    public NDArray rdiv(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray rdivi(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray rsub(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray rsubi(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray div(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray divi(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray mul(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray muli(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray sub(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray subi(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray match(NDArray comp, Condition condition) {
        return null;
    }

    @Override
    public NDArray match(Number comp, Condition condition) {
        return null;
    }

    @Override
    public NDArray getWhere(NDArray comp, Condition condition) {
        return null;
    }

    @Override
    public NDArray getWhere(Number comp, Condition condition) {
        return null;
    }

    @Override
    public NDArray putWhere(NDArray comp, NDArray put, Condition condition) {
        return null;
    }

    @Override
    public NDArray putWhere(Number comp, NDArray put, Condition condition) {
        return null;
    }

    @Override
    public NDArray putWhereWithMask(NDArray mask, NDArray put) {
        return null;
    }

    @Override
    public NDArray putWhereWithMask(NDArray mask, Number put) {
        return null;
    }

    @Override
    public NDArray putWhere(Number comp, Number put, Condition condition) {
        return null;
    }

    @Override
    public NDArray get(NDArray indices) {
        return null;
    }

    @Override
    public NDArray get(List<List<Integer>> indices) {
        return null;
    }

    @Override
    public NDArray rdiv(NDArray other) {
        return null;
    }

    @Override
    public NDArray rdivi(NDArray other) {
        return null;
    }

    @Override
    public NDArray rdiv(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray rdivi(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray rsub(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray rsub(NDArray other) {
        return null;
    }

    @Override
    public NDArray rsubi(NDArray other) {
        return null;
    }

    @Override
    public NDArray rsubi(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray assign(Number value) {
        return null;
    }

    @Override
    public NDArray putSlice(int slice, NDArray put) {
        return null;
    }

    @Override
    public NDArray cond(Condition condition) {
        return null;
    }

    @Override
    public NDArray tile(int repeats) {
        return null;
    }

    @Override
    public NDArray tile(int axis, int repeats) {
        return null;
    }

    @Override
    public NDArray tile(int[] repeats) {
        return null;
    }

    @Override
    public NDArray tile(Shape desiredShape) {
        return null;
    }

    @Override
    public NDArray repeat(int repeats) {
        return null;
    }

    @Override
    public NDArray repeat(int axis, int repeats) {
        return null;
    }

    @Override
    public NDArray repeat(int[] repeats) {
        return null;
    }

    @Override
    public NDArray repeat(Shape desiredShape) {
        return null;
    }

    @Override
    public NDArray getScalar(long i) {
        return null;
    }

    @Override
    public NDArray put(List<List<Integer>> indices, NDArray element) {
        return null;
    }

    @Override
    public NDArray put(NDArray indices, NDArray element) {
        return null;
    }

    @Override
    public NDArray put(NDArray element, int... indices) {
        return null;
    }

    @Override
    public NDArray put(int i, NDArray element) {
        return null;
    }

    @Override
    public NDArray mmul(NDArray other) {
        return null;
    }

    @Override
    public double[] toDoubleArray() {
        return new double[0];
    }

    @Override
    public float[] toFloatArray() {
        return new float[0];
    }

    @Override
    public int[] toIntArray() {
        return new int[0];
    }

    @Override
    public long[] toLongArray() {
        return new long[0];
    }

    @Override
    public NDArray mmul(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray div(NDArray other) {
        return null;
    }

    @Override
    public NDArray div(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray mul(NDArray other) {
        return null;
    }

    @Override
    public NDArray mul(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray sub(NDArray other) {
        return null;
    }

    @Override
    public NDArray sub(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray add(NDArray other) {
        return null;
    }

    @Override
    public NDArray mmuli(NDArray other) {
        return null;
    }

    @Override
    public NDArray mmuli(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray divi(NDArray other) {
        return null;
    }

    @Override
    public NDArray divi(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray muli(NDArray other) {
        return null;
    }

    @Override
    public NDArray muli(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray subi(NDArray other) {
        return null;
    }

    @Override
    public NDArray subi(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray amax(int... dimension) {
        return null;
    }

    @Override
    public Number amaxNumber() {
        return null;
    }

    @Override
    public NDArray amin(int... dimension) {
        return null;
    }

    @Override
    public Number aminNumber() {
        return null;
    }

    @Override
    public Number max() {
        return null;
    }

    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        return null;
    }

    @Override
    public Number min() {
        return null;
    }

    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        return null;
    }

    @Override
    public Number sum() {
        return null;
    }

    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        return null;
    }

    @Override
    public Number prod() {
        return null;
    }

    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        return null;
    }

    @Override
    public Number mean() {
        return null;
    }

    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        return null;
    }

    @Override
    public NDArray getScalar(int... indices) {
        return null;
    }

    @Override
    public NDArray getScalar(long... indices) {
        return null;
    }

    @Override
    public long getLong(int... indices) {
        return 0;
    }

    @Override
    public long getLong(long... indices) {
        return 0;
    }

    @Override
    public double getDouble(int... indices) {
        return 0;
    }

    @Override
    public double getDouble(long... indices) {
        return 0;
    }

    @Override
    public float getFloat(int... indices) {
        return 0;
    }

    @Override
    public float getFloat(long... indices) {
        return 0;
    }

    @Override
    public NDArray dup() {
        return null;
    }

    @Override
    public NDArray slice(long i, int dimension) {
        return null;
    }

    @Override
    public NDArray slice(long i) {
        return null;
    }

    @Override
    public NDArray flatten() {
        return null;
    }

    @Override
    public NDArray reshape(Shape shape) {
        return null;
    }

    @Override
    public NDArray swapAxes(int dimension, int with) {
        return null;
    }

    @Override
    public NDArray transpose(int... dimensions) {
        return null;
    }

    @Override
    public NDArray transposei(int... dimensions) {
        return null;
    }

    @Override
    public NDArray expandDims(int axis) {
        return null;
    }

    @Override
    public NDArray stack(NDArray[] arrays, int axis) {
        return null;
    }

    @Override
    public NDArray stack(NDList arrays, int axis) {
        return null;
    }

    @Override
    public NDArray concat(NDArray[] arrays, int axis) {
        return null;
    }

    @Override
    public NDArray concat(NDList arrays, int axis) {
        return null;
    }

    @Override
    public NDArray clip(double min, double max) {
        return null;
    }

    @Override
    public long size(int dimension) {
        return 0;
    }

    @Override
    public long size() {
        return 0;
    }

    @Override
    public NDArray broadcast(long... shape) {
        return null;
    }

    @Override
    public NDArray broadcast(NDArray result) {
        return null;
    }

    @Override
    public Object element() {
        return null;
    }

    @Override
    public boolean equalsWithEps(Object o, double eps) {
        return false;
    }

    @Override
    public boolean equalShapes(NDArray other) {
        return false;
    }

    @Override
    public NDArray remainder(NDArray denominator) {
        return null;
    }

    @Override
    public NDArray remainder(NDArray denominator, NDArray result) {
        return null;
    }

    @Override
    public NDArray remainder(Number denominator) {
        return null;
    }

    @Override
    public NDArray remainder(Number denominator, NDArray result) {
        return null;
    }

    @Override
    public NDArray remainderi(NDArray denominator) {
        return null;
    }

    @Override
    public NDArray remainderi(Number denominator) {
        return null;
    }

    @Override
    public NDArray fmod(NDArray denominator) {
        return null;
    }

    @Override
    public NDArray fmod(NDArray denominator, NDArray result) {
        return null;
    }

    @Override
    public NDArray fmod(Number denominator) {
        return null;
    }

    @Override
    public NDArray fmod(Number denominator, NDArray result) {
        return null;
    }

    @Override
    public NDArray fmodi(NDArray denominator) {
        return null;
    }

    @Override
    public NDArray fmodi(Number denominator) {
        return null;
    }

    @Override
    public NDArray argMax(int... dimension) {
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
    public int nonzero() {
        return 0;
    }

    @Override
    public boolean isEmpty() {
        return false;
    }

    @Override
    public NDArray castTo(DataType dataType) {
        return null;
    }

    @Override
    public Matrix asMatrix() {
        return null;
    }

    @Override
    public boolean all() {
        return false;
    }

    @Override
    public boolean any() {
        return false;
    }

    @Override
    public boolean none() {
        return false;
    }

    @Override
    public NDArray like() {
        return null;
    }

    @Override
    public NDArray ulike() {
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

    public void close() {}
}
