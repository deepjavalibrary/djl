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

import java.nio.Buffer;
import java.nio.ByteBuffer;
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

public class MockNDArray implements NDArray {

    @Override
    public NDManager getManager() {
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
    public DataDesc getDataDescriptor() {
        return null;
    }

    @Override
    public SparseFormat getSparseFormat() {
        return null;
    }

    @Override
    public boolean isSparse() {
        return false;
    }

    @Override
    public NDArray asInContext(Context ctx, boolean copy) {
        return null;
    }

    @Override
    public NDArray asType(DataType dtype, boolean copy) {
        return null;
    }

    @Override
    public Matrix asMatrix() {
        return null;
    }

    @Override
    public void backward() {}

    @Override
    public void backward(boolean retainGraph, boolean isTraining) {}

    @Override
    public void backward(NDArray outGrad, boolean retainGraph, boolean isTraining) {}

    @Override
    public void attachGradient() {}

    @Override
    public void attachGradient(GradReq gradReq, SparseFormat sparseFormat) {}

    @Override
    public NDArray getGradient() {
        return null;
    }

    @Override
    public ByteBuffer toByteBuffer() {
        return null;
    }

    @Override
    public void set(Buffer data) {}

    @Override
    public NDArray set(NDIndex index, NDArray value) {
        return null;
    }

    @Override
    public NDArray set(NDIndex index, Number value) {
        return null;
    }

    @Override
    public NDArray setElement(NDIndex index, Number value) {
        return null;
    }

    @Override
    public NDArray seti(NDIndex index, NDArray value) {
        return null;
    }

    @Override
    public NDArray seti(NDIndex index, Number value) {
        return null;
    }

    @Override
    public NDArray setElementi(NDIndex index, Number value) {
        return null;
    }

    @Override
    public NDArray get(NDIndex index) {
        return null;
    }

    @Override
    public void copyTo(NDArray array) {}

    @Override
    public NDArray dup() {
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
    public NDArray like() {
        return null;
    }

    @Override
    public boolean contentEquals(Number number) {
        return false;
    }

    @Override
    public boolean contentEquals(NDArray other) {
        return false;
    }

    @Override
    public boolean equalsWithEps(Object o, double eps) {
        return false;
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
    public NDArray eps(Number other) {
        return null;
    }

    @Override
    public NDArray eps(NDArray other) {
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
    public NDArray gt(Number other) {
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
    public NDArray lt(Number other) {
        return null;
    }

    @Override
    public NDArray lt(NDArray other) {
        return null;
    }

    @Override
    public NDArray lte(Number other) {
        return null;
    }

    @Override
    public NDArray lte(NDArray other) {
        return null;
    }

    @Override
    public NDArray add(Number n) {
        return null;
    }

    @Override
    public NDArray add(NDArray... others) {
        return null;
    }

    @Override
    public NDArray sub(Number n) {
        return null;
    }

    @Override
    public NDArray sub(NDArray other) {
        return null;
    }

    @Override
    public NDArray mul(Number n) {
        return null;
    }

    @Override
    public NDArray mul(NDArray... others) {
        return null;
    }

    @Override
    public NDArray div(Number n) {
        return null;
    }

    @Override
    public NDArray div(NDArray other) {
        return null;
    }

    @Override
    public NDArray mod(Number n) {
        return null;
    }

    @Override
    public NDArray mod(NDArray other) {
        return null;
    }

    @Override
    public NDArray pow(Number n) {
        return null;
    }

    @Override
    public NDArray pow(NDArray other) {
        return null;
    }

    @Override
    public NDArray addi(Number n) {
        return null;
    }

    @Override
    public NDArray addi(NDArray... others) {
        return null;
    }

    @Override
    public NDArray subi(Number n) {
        return null;
    }

    @Override
    public NDArray subi(NDArray other) {
        return null;
    }

    @Override
    public NDArray muli(Number n) {
        return null;
    }

    @Override
    public NDArray muli(NDArray... others) {
        return null;
    }

    @Override
    public NDArray divi(Number n) {
        return null;
    }

    @Override
    public NDArray divi(NDArray other) {
        return null;
    }

    @Override
    public NDArray modi(Number n) {
        return null;
    }

    @Override
    public NDArray modi(NDArray other) {
        return null;
    }

    @Override
    public NDArray powi(Number n) {
        return null;
    }

    @Override
    public NDArray powi(NDArray other) {
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
    public NDArray toDegrees() {
        return null;
    }

    @Override
    public NDArray toRadians() {
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
    public NDArray trace(int offset, int axis1, int axis2) {
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
    public NDArray flatten() {
        return null;
    }

    @Override
    public NDArray reshape(Shape shape) {
        return null;
    }

    @Override
    public NDArray expandDims(int axis) {
        return null;
    }

    @Override
    public NDArray squeeze(int[] axes) {
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
    public NDArray argsort(int axis, boolean ascending) {
        return null;
    }

    @Override
    public NDArray sort(int axis) {
        return null;
    }

    @Override
    public NDArray sort() {
        return null;
    }

    @Override
    public NDArray softmax(int[] axes, double temperature) {
        return null;
    }

    @Override
    public NDArray cumsumi(int axis) {
        return null;
    }

    @Override
    public NDArray cumsumi() {
        return null;
    }

    @Override
    public NDArray cumsum(int axis) {
        return null;
    }

    @Override
    public NDArray cumsum() {
        return null;
    }

    @Override
    public NDArray toSparse(SparseFormat fmt) {
        return null;
    }

    @Override
    public NDArray toDense() {
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
    public NDArray createMask(NDIndex index) {
        return null;
    }

    @Override
    public NDArray createMask(Predicate<Number> predicate) {
        return null;
    }

    @Override
    public NDArray tile(long repeats) {
        return null;
    }

    @Override
    public NDArray tile(int axis, long repeats) {
        return null;
    }

    @Override
    public NDArray tile(long[] repeats) {
        return null;
    }

    @Override
    public NDArray tile(Shape desiredShape) {
        return null;
    }

    @Override
    public NDArray repeat(long repeats) {
        return null;
    }

    @Override
    public NDArray repeat(int axis, long repeats) {
        return null;
    }

    @Override
    public NDArray repeat(long[] repeats) {
        return null;
    }

    @Override
    public NDArray repeat(Shape desiredShape) {
        return null;
    }

    @Override
    public NDArray mmul(NDArray other) {
        return null;
    }

    @Override
    public NDArray clip(double min, double max) {
        return null;
    }

    @Override
    public NDArray transpose() {
        return null;
    }

    @Override
    public NDArray transpose(int[] dimensions) {
        return null;
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
    public boolean equalShapes(NDArray other) {
        return false;
    }

    @Override
    public NDArray argmax() {
        return null;
    }

    @Override
    public NDArray argmax(int axis, boolean keepDims) {
        return null;
    }

    @Override
    public NDArray argmin() {
        return null;
    }

    @Override
    public NDArray argmin(int axis, boolean keepDims) {
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
    public long nonzero() {
        return 0;
    }

    @Override
    public boolean isEmpty() {
        return false;
    }

    @Override
    public NDArray logicalNot() {
        return null;
    }

    @Override
    public NDArrayEx getNDArrayInternal() {
        return null;
    }

    @Override
    public void close() {}
}
