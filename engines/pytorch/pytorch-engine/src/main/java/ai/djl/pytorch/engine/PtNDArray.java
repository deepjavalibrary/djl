/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.engine;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.util.NativeResource;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;

/** {@code PtNDArray} is the interface for the PyTorch implementation of {@link NDArray}. */
public interface PtNDArray extends NativeResource<Long>, NDArray {

    /** {@inheritDoc} */
    @Override
    PtNDManager getManager();

    /** {@inheritDoc} */
    @Override
    String getName();

    /** {@inheritDoc} */
    @Override
    void setName(String name);

    /** {@inheritDoc} */
    @Override
    DataType getDataType();

    /** {@inheritDoc} */
    @Override
    Device getDevice();

    /** {@inheritDoc} */
    @Override
    Shape getShape();

    /** {@inheritDoc} */
    @Override
    SparseFormat getSparseFormat();

    /** {@inheritDoc} */
    @Override
    PtNDArray toDevice(Device device, boolean copy);

    /** {@inheritDoc} */
    @Override
    PtNDArray toType(DataType dataType, boolean copy);

    /** {@inheritDoc} */
    @Override
    void setRequiresGradient(boolean requiresGrad);

    /** {@inheritDoc} */
    @Override
    PtNDArray getGradient();

    /** {@inheritDoc} */
    @Override
    boolean hasGradient();

    /** {@inheritDoc} */
    @Override
    NDArray stopGradient();

    /** {@inheritDoc} */
    @Override
    ByteBuffer toByteBuffer();

    /** {@inheritDoc} */
    @Override
    String[] toStringArray(Charset charset);

    /** {@inheritDoc} */
    @Override
    void set(Buffer buffer);

    /** {@inheritDoc} */
    @Override
    NDArray get(NDManager manager, long... indices);

    /** {@inheritDoc} */
    @Override
    NDArray gather(NDArray index, int axis);

    /** {@inheritDoc} */
    @Override
    NDArray gatherNd(NDArray index);

    /** {@inheritDoc} */
    @Override
    NDArray take(NDManager manager, NDArray index);

    /** {@inheritDoc} */
    @Override
    NDArray put(NDArray index, NDArray data);

    /** {@inheritDoc} */
    @Override
    void copyTo(NDArray array);

    /** {@inheritDoc} */
    @Override
    void attach(NDManager manager);

    /** {@inheritDoc} */
    @Override
    void returnResource(NDManager manager);

    /** {@inheritDoc} */
    @Override
    void tempAttach(NDManager manager);

    /** {@inheritDoc} */
    @Override
    void detach();

    /** {@inheritDoc} */
    @Override
    NDArray duplicate();

    /** {@inheritDoc} */
    @Override
    PtNDArray booleanMask(NDArray index, int axis);

    /** {@inheritDoc} */
    @Override
    NDArray sequenceMask(NDArray sequenceLength, float value);

    /** {@inheritDoc} */
    @Override
    NDArray sequenceMask(NDArray sequenceLength);

    /** {@inheritDoc} */
    @Override
    boolean contentEquals(Number number);

    /** {@inheritDoc} */
    @Override
    boolean contentEquals(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray eq(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray eq(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray neq(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray neq(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray gt(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray gt(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray gte(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray gte(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray lt(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray lt(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray lte(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray lte(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray add(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray add(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray sub(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray sub(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray mul(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray mul(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray div(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray div(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray mod(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray mod(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray pow(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray pow(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray addi(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray addi(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray subi(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray subi(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray muli(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray muli(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray divi(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray divi(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray modi(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray modi(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray powi(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray powi(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray sign();

    /** {@inheritDoc} */
    @Override
    PtNDArray signi();

    /** {@inheritDoc} */
    @Override
    PtNDArray maximum(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray maximum(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray minimum(Number n);

    /** {@inheritDoc} */
    @Override
    PtNDArray minimum(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray all();

    /** {@inheritDoc} */
    @Override
    PtNDArray any();

    /** {@inheritDoc} */
    @Override
    PtNDArray none();

    /** {@inheritDoc} */
    @Override
    PtNDArray neg();

    /** {@inheritDoc} */
    @Override
    PtNDArray negi();

    /** {@inheritDoc} */
    @Override
    PtNDArray abs();

    /** {@inheritDoc} */
    @Override
    PtNDArray square();

    /** {@inheritDoc} */
    @Override
    NDArray sqrt();

    /** {@inheritDoc} */
    @Override
    PtNDArray cbrt();

    /** {@inheritDoc} */
    @Override
    PtNDArray floor();

    /** {@inheritDoc} */
    @Override
    PtNDArray ceil();

    /** {@inheritDoc} */
    @Override
    PtNDArray round();

    /** {@inheritDoc} */
    @Override
    PtNDArray trunc();

    /** {@inheritDoc} */
    @Override
    PtNDArray exp();

    /** {@inheritDoc} */
    @Override
    NDArray gammaln();

    /** {@inheritDoc} */
    @Override
    PtNDArray log();

    /** {@inheritDoc} */
    @Override
    PtNDArray log10();

    /** {@inheritDoc} */
    @Override
    PtNDArray log2();

    /** {@inheritDoc} */
    @Override
    PtNDArray sin();

    /** {@inheritDoc} */
    @Override
    PtNDArray cos();

    /** {@inheritDoc} */
    @Override
    PtNDArray tan();

    /** {@inheritDoc} */
    @Override
    PtNDArray asin();

    /** {@inheritDoc} */
    @Override
    PtNDArray acos();

    /** {@inheritDoc} */
    @Override
    PtNDArray atan();

    /** {@inheritDoc} */
    @Override
    PtNDArray sinh();

    /** {@inheritDoc} */
    @Override
    PtNDArray cosh();

    /** {@inheritDoc} */
    @Override
    PtNDArray tanh();

    /** {@inheritDoc} */
    @Override
    PtNDArray asinh();

    /** {@inheritDoc} */
    @Override
    PtNDArray acosh();

    /** {@inheritDoc} */
    @Override
    PtNDArray atanh();

    /** {@inheritDoc} */
    @Override
    PtNDArray toDegrees();

    /** {@inheritDoc} */
    @Override
    PtNDArray toRadians();

    /** {@inheritDoc} */
    @Override
    PtNDArray max();

    /** {@inheritDoc} */
    @Override
    PtNDArray max(int[] axes, boolean keepDims);

    /** {@inheritDoc} */
    @Override
    PtNDArray min();

    /** {@inheritDoc} */
    @Override
    PtNDArray min(int[] axes, boolean keepDims);

    /** {@inheritDoc} */
    @Override
    PtNDArray sum();

    /** {@inheritDoc} */
    @Override
    PtNDArray sum(int[] axes, boolean keepDims);

    /** {@inheritDoc} */
    @Override
    NDArray cumProd(int axis);

    /** {@inheritDoc} */
    @Override
    NDArray cumProd(int axis, DataType dataType);

    /** {@inheritDoc} */
    @Override
    PtNDArray prod();

    /** {@inheritDoc} */
    @Override
    PtNDArray prod(int[] axes, boolean keepDims);

    /** {@inheritDoc} */
    @Override
    PtNDArray mean();

    /** {@inheritDoc} */
    @Override
    PtNDArray mean(int[] axes, boolean keepDims);

    /** {@inheritDoc} */
    @Override
    PtNDArray normalize(double p, long dim, double eps);

    /** {@inheritDoc} */
    @Override
    PtNDArray rotate90(int times, int[] axes);

    /** {@inheritDoc} */
    @Override
    PtNDArray trace(int offset, int axis1, int axis2);

    /** {@inheritDoc} */
    @Override
    NDList split(long sections, int axis);

    /** {@inheritDoc} */
    @Override
    NDList split(long[] indices, int axis);

    /** {@inheritDoc} */
    @Override
    PtNDArray flatten();

    /** {@inheritDoc} */
    @Override
    NDArray flatten(int startDim, int endDim);

    /** {@inheritDoc} */
    @Override
    PtNDArray reshape(Shape shape);

    /** {@inheritDoc} */
    @Override
    PtNDArray expandDims(int axis);

    /** {@inheritDoc} */
    @Override
    PtNDArray squeeze();

    /** {@inheritDoc} */
    @Override
    PtNDArray squeeze(int axis);

    /** {@inheritDoc} */
    @Override
    PtNDArray squeeze(int[] axes);

    /** {@inheritDoc} */
    @Override
    PtNDArray logicalAnd(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray logicalOr(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray logicalXor(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray logicalNot();

    /** {@inheritDoc} */
    @Override
    PtNDArray argSort(int axis, boolean ascending);

    /** {@inheritDoc} */
    @Override
    PtNDArray sort();

    /** {@inheritDoc} */
    @Override
    PtNDArray sort(int axis);

    /** {@inheritDoc} */
    @Override
    PtNDArray softmax(int axis);

    /** {@inheritDoc} */
    @Override
    PtNDArray logSoftmax(int axis);

    /** {@inheritDoc} */
    @Override
    PtNDArray cumSum();

    /** {@inheritDoc} */
    @Override
    PtNDArray cumSum(int axis);

    /** {@inheritDoc} */
    @Override
    void intern(NDArray replaced);

    /** {@inheritDoc} */
    @Override
    PtNDArray isInfinite();

    /** {@inheritDoc} */
    @Override
    PtNDArray isNaN();

    /** {@inheritDoc} */
    @Override
    PtNDArray tile(long repeats);

    /** {@inheritDoc} */
    @Override
    PtNDArray tile(int axis, long repeats);

    /** {@inheritDoc} */
    @Override
    PtNDArray tile(long[] repeats);

    /** {@inheritDoc} */
    @Override
    PtNDArray tile(Shape desiredShape);

    /** {@inheritDoc} */
    @Override
    PtNDArray repeat(long repeats);

    /** {@inheritDoc} */
    @Override
    PtNDArray repeat(int axis, long repeats);

    /** {@inheritDoc} */
    @Override
    PtNDArray repeat(long[] repeats);

    /** {@inheritDoc} */
    @Override
    PtNDArray repeat(Shape desiredShape);

    /** {@inheritDoc} */
    @Override
    PtNDArray dot(NDArray other);

    /** {@inheritDoc} */
    @Override
    NDArray matMul(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArray clip(Number min, Number max);

    /** {@inheritDoc} */
    @Override
    PtNDArray swapAxes(int axis1, int axis2);

    /** {@inheritDoc} */
    @Override
    NDArray flip(int... axes);

    /** {@inheritDoc} */
    @Override
    PtNDArray transpose();

    /** {@inheritDoc} */
    @Override
    PtNDArray transpose(int... axes);

    /** {@inheritDoc} */
    @Override
    PtNDArray broadcast(Shape shape);

    /** {@inheritDoc} */
    @Override
    PtNDArray argMax();

    /** {@inheritDoc} */
    @Override
    PtNDArray argMax(int axis);

    /** {@inheritDoc} */
    @Override
    PtNDArray argMin();

    /** {@inheritDoc} */
    @Override
    PtNDArray argMin(int axis);

    /** {@inheritDoc} */
    @Override
    PtNDArray percentile(Number percentile);

    /** {@inheritDoc} */
    @Override
    PtNDArray percentile(Number percentile, int[] axes);

    /** {@inheritDoc} */
    @Override
    PtNDArray median();

    /** {@inheritDoc} */
    @Override
    PtNDArray median(int[] axes);

    /** {@inheritDoc} */
    @Override
    PtNDArray toDense();

    /** {@inheritDoc} */
    @Override
    PtNDArray toSparse(SparseFormat fmt);

    /** {@inheritDoc} */
    @Override
    PtNDArray nonzero();

    /** {@inheritDoc} */
    @Override
    PtNDArray erfinv();

    /** {@inheritDoc} */
    @Override
    PtNDArray inverse();

    /** {@inheritDoc} */
    @Override
    NDArray norm(boolean keepDims);

    /** {@inheritDoc} */
    @Override
    NDArray norm(int order, int[] axes, boolean keepDims);

    /** {@inheritDoc} */
    @Override
    NDArray oneHot(int depth);

    /** {@inheritDoc} */
    @Override
    NDArray oneHot(int depth, DataType dataType);

    /** {@inheritDoc} */
    @Override
    NDArray oneHot(int depth, float onValue, float offValue, DataType dataType);

    /** {@inheritDoc} */
    @Override
    NDArray batchDot(NDArray other);

    /** {@inheritDoc} */
    @Override
    PtNDArrayEx getNDArrayInternal();

    /** {@inheritDoc} */
    @Override
    String toString();

    /** {@inheritDoc} */
    @Override
    boolean equals(Object obj);

    /** {@inheritDoc} */
    @Override
    int hashCode();

    /** {@inheritDoc} */
    @Override
    void close();
}
