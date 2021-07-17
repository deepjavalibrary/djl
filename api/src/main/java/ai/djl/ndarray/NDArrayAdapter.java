/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.ndarray;

import ai.djl.Device;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import java.nio.Buffer;

/**
 * A base implementation of the {@link NDArray} that does nothing. This can be used for overriding
 * the NDArray with only part of the interface implemented.
 *
 * <p>This interface should only be used for the NDArray implementations that do not plan to
 * implement a large portion of the interface. For the ones that do, they should directly implement
 * {@link NDArray} so that the unsupported operations are better highlighted in the code.
 */
public interface NDArrayAdapter extends NDArray {

    String UNSUPPORTED_MSG =
            "This NDArray implementation does not currently support this operation";

    /** {@inheritDoc} */
    @Override
    default SparseFormat getSparseFormat() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray toDevice(Device device, boolean copy) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray toType(DataType dataType, boolean copy) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default void setRequiresGradient(boolean requiresGrad) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray getGradient() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default boolean hasGradient() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray stopGradient() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default String[] toStringArray() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default void set(Buffer data) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default void copyTo(NDArray array) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray booleanMask(NDArray index, int axis) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray sequenceMask(NDArray sequenceLength, float value) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray sequenceMask(NDArray sequenceLength) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray zerosLike() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray onesLike() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default boolean contentEquals(Number number) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default boolean contentEquals(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray eq(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray eq(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray neq(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray neq(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray gt(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray gt(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray gte(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray gte(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray lt(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray lt(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray lte(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray lte(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray add(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray add(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray sub(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray sub(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray mul(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray mul(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray div(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray div(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray mod(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray mod(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray pow(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray pow(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray addi(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray addi(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray subi(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray subi(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray muli(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray muli(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray divi(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray divi(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray modi(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray modi(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray powi(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray powi(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray sign() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray signi() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray maximum(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray maximum(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray minimum(Number n) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray minimum(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray neg() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray negi() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray abs() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray square() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray sqrt() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray cbrt() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray floor() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray ceil() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray round() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray trunc() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray exp() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray log() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray log10() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray log2() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray sin() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray cos() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray tan() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray asin() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray acos() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray atan() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray sinh() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray cosh() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray tanh() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray asinh() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray acosh() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray atanh() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray toDegrees() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray toRadians() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray max() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray max(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray min() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray min(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray sum() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray sum(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray prod() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray prod(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray mean() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray mean(int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray rotate90(int times, int[] axes) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray trace(int offset, int axis1, int axis2) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDList split(long[] indices, int axis) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray flatten() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray reshape(Shape shape) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray expandDims(int axis) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray squeeze(int[] axes) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray logicalAnd(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray logicalOr(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray logicalXor(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray logicalNot() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray argSort(int axis, boolean ascending) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray sort() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray sort(int axis) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray softmax(int axis) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray logSoftmax(int axis) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray cumSum() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray cumSum(int axis) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default void intern(NDArray replaced) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray isInfinite() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray isNaN() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray tile(long repeats) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray tile(int axis, long repeats) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray tile(long[] repeats) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray tile(Shape desiredShape) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray repeat(long repeats) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray repeat(int axis, long repeats) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray repeat(long[] repeats) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray repeat(Shape desiredShape) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray dot(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray matMul(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray clip(Number min, Number max) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray flip(int... axes) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray transpose() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray transpose(int... axes) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray broadcast(Shape shape) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray argMax() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray argMax(int axis) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray argMin() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray argMin(int axis) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray percentile(Number percentile) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray percentile(Number percentile, int[] axes) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray median() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray median(int[] axes) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray toDense() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray toSparse(SparseFormat fmt) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray nonzero() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray erfinv() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray norm(boolean keepDims) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray norm(int ord, int[] axes, boolean keepDims) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray oneHot(int depth, float onValue, float offValue, DataType dataType) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArray batchDot(NDArray other) {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }

    /** {@inheritDoc} */
    @Override
    default NDArrayEx getNDArrayInternal() {
        throw new UnsupportedOperationException(UNSUPPORTED_MSG);
    }
}
