/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.engine.rust;

import java.nio.ByteBuffer;

/** Rust native library. */
@SuppressWarnings({"unused", "MissingJavadocMethod"})
public final class RustLibrary {

    private RustLibrary() {}

    public static native boolean isCudaAvailable();

    public static native long loadModel(
            String modelPath, int dtype, String deviceType, int deviceId);

    public static native long deleteModel(long handle);

    public static native String[] getInputNames(long handle);

    public static native long runInference(long handle, long[] inputHandles);

    public static native long tensorOf(
            ByteBuffer buf, long[] shape, int dataType, String deviceType, int deviceId);

    public static native long zeros(long[] shape, int dataType, String deviceType, int deviceId);

    public static native long ones(long[] shape, int dType, String deviceType, int deviceId);

    public static native long full(
            float value, long[] shape, int dataType, String deviceType, int deviceId);

    public static native long arange(
            float start, float stop, float step, int dataType, String deviceType, int deviceId);

    public static native long eye(
            int rows, int cols, int dataType, String deviceType, int deviceId);

    public static long linspace(
            float start, float stop, int num, int dataType, String deviceType, int deviceId) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long randint(
            long low, long high, long[] shape, int dataType, String deviceType, int deviceId) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long randomPermutation(long n, String deviceType, int deviceId) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long uniform(
            float low, float high, long[] shape, int dataType, String deviceType, int deviceId);

    public static native long randomNormal(
            float loc, float scale, long[] shape, int dataType, String deviceType, int deviceId);

    public static long hannWindow(long numPoints, String deviceType, int deviceId) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native void deleteTensor(long handle);

    public static native int getDataType(long handle);

    public static native int[] getDevice(long handle);

    public static native long[] getShape(long handle);

    public static native long duplicate(long handle);

    public static native long toDevice(long handle, String deviceType, int deviceId);

    public static native long toBoolean(long handle);

    public static native long toDataType(long handle, int dataType);

    public static native byte[] toByteArray(long handle);

    public static native long fullSlice(long handle, long[] min, long[] max, long[] step);

    public static native long gather(long handle, long indexHandle, int axis);

    public static long take(long handle, long indexHandle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long put(long handle, long indexHandle, long valueHandle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long scatter(long handle, long indexHandle, long valueHandle, int axis);

    public static long booleanMask(long handle, long indexHandle, int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    // comparison ops

    public static native boolean contentEqual(long handle, long other);

    public static native long eq(long handle, long other);

    public static native long neq(long handle, long other);

    public static native long gt(long handle, long other);

    public static native long gte(long handle, long other);

    public static native long lt(long handle, long other);

    public static native long lte(long handle, long other);

    // binary  ops

    public static native long add(long handle, long other);

    public static native long sub(long handle, long other);

    public static native long mul(long handle, long other);

    public static native long div(long handle, long other);

    public static native long minimum(long handle, long other);

    public static native long maximum(long handle, long other);

    public static long remainder(long handle, long other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long pow(long handle, long other);

    public static long xlogy(long handle, long other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    // unary ops

    public static native long exp(long handle);

    public static native long log(long handle);

    public static long log10(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long log2(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long sin(long handle);

    public static native long cos(long handle);

    public static long tan(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long asin(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long acos(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long atan(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long atan2(long handle, long other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long sinh(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long cosh(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long tanh(long handle);

    public static native long abs(long handle);

    public static native long neg(long handle);

    public static long sign(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long square(long handle);

    public static native long sqrt(long handle);

    public static native long floor(long handle);

    public static native long ceil(long handle);

    public static native long round(long handle);

    public static long trunc(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long countNonzero(long handle);

    public static native long countNonzeroWithAxis(long handle, int axis);

    // reduce ops

    public static native long sum(long handle);

    public static native long sumWithAxis(long handle, int[] axes, boolean keepDims);

    public static long[] topK(long handle, int k, int axis, boolean largest, boolean sorted) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long max(long handle);

    public static native long maxWithAxis(long handle, int axis, boolean keepDims);

    public static native long min(long handle);

    public static native long minWithAxis(long handle, int axis, boolean keepDims);

    public static native long argMax(long handle);

    public static native long argMaxWithAxis(long handle, int axis, boolean keepDims);

    public static native long argMin(long handle);

    public static native long argMinWithAxis(long handle, int axis, boolean keepDims);

    public static long percentile(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long percentileWithAxes(long handle, double percentile, int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long mean(long handle);

    public static native long meanWithAxis(long handle, int[] axis, boolean keepDims);

    public static long[] median(long handle, int axis, boolean keepDims) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long cumProd(long handle, int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long cumProdWithType(long handle, int axis, int dataType) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long prod(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long cumProdWithAxis(long handle, int axis, boolean keepDims) {
        throw new UnsupportedOperationException("Not implemented");
    }

    // other ops

    public static native long normalize(long handle, double p, long dim, double eps);

    public static long rot90(long handle, int times, int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long[] split(long handle, long[] indices, int axis);

    public static native long flatten(long handle);

    public static native long flattenWithDims(long handle, int startDim, int endDim);

    public static native long reshape(long handle, long[] shape);

    public static native long expandDims(long handle, int axis);

    public static native long squeeze(long handle, int[] axes);

    public static long logicalAnd(long handle, long other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long logicalOr(long handle, long other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long logicalXor(long handle, long other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long logicalNot(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long argSort(long handle, int axis, boolean ascending) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long sort(long handle, int axis, boolean ascending) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long softmax(long handle, int axis);

    public static native long logSoftmax(long handle, int axis);

    public static native long cumSum(long handle, int axis);

    public static long isInf(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long isNaN(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long tile(long handle, long[] repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long tileWithAxis(long handle, int axis, long repeat) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long tileWithShape(long handle, long[] shape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long repeat(long handle, long repeat, int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long dot(long handle, long other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long matmul(long handle, long other);

    public static native long batchMatMul(long handle, long other);

    public static native long clip(long handle, double min, double max);

    public static native long transpose(long handle, int axis1, int axis2);

    public static long flip(long handle, int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long permute(long handle, int[] axes);

    public static native long broadcast(long handle, long[] shape);

    public static long nonZero(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long inverse(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long norm(long handle, int i, int[] ints, boolean keepDims) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long oneHot(long handle, int depth, float onValue, float offValue, int dataType) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long complex(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long real(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long sigmoid(long handle);

    public static long softPlus(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long softSign(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long leakyRelu(long handle, float alpha);

    public static long elu(long handle, float alpha) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long selu(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long relu(long handle);

    public static native long gelu(long handle);

    public static native long erf(long handle);

    public static long erfinv(long handle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long maxPool(
            long handle, long[] kernelShape, long[] stride, long[] padding, boolean ceilMode) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long adaptiveMaxPool(long handle, long[] shape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long avgPool2d(long handle, long[] kernelShape, long[] stride);

    public static long adaptiveAvgPool(long handle, long[] shape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long lpPool(
            long handle, float normType, long[] kernelShape, long[] stride, boolean ceilMode) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static long where(long conditionHandle, long handle, long otherHandle) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public static native long stack(long[] srcArray, int axis);

    public static native long concat(long[] srcArray, int axis);

    public static long pick(long handle, long pickHandle, int axis) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
