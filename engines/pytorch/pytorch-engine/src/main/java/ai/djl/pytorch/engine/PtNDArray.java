/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.util.NativeResource;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/** {@code PtNDArray} is the PyTorch implementation of {@link NDArray}. */
public class PtNDArray extends NativeResource<Long> implements NDArray {

    private String name;
    private Device device;
    private DataType dataType;
    private Shape shape;
    private SparseFormat sparseFormat;
    // use Boolean object to maintain three status: null, false, true
    private Boolean hasGradient;
    private PtNDManager manager;
    private PtNDArrayEx ptNDArrayEx;

    // keep a reference to direct buffer to avoid GC release the memory
    @SuppressWarnings("PMD.UnusedPrivateField")
    private ByteBuffer dataRef;

    /**
     * Constructs a PyTorch {@code NDArray} from a native handle (internal. Use {@link NDManager}
     * instead).
     *
     * @param manager the manager to attach the new array to
     * @param handle the pointer to the native PyTorch memory
     */
    public PtNDArray(PtNDManager manager, long handle) {
        super(handle);
        this.manager = manager;
        this.ptNDArrayEx = new PtNDArrayEx(this);
        manager.attachInternal(getUid(), this);
    }

    /**
     * Constructs a PyTorch {@code NDArray} from a native handle (internal. Use {@link NDManager}
     * instead) with the data that is hold on Java side.
     *
     * @param manager the manager to attach the new array to
     * @param handle the pointer to the native PyTorch memory
     * @param data the direct buffer of the data
     */
    public PtNDArray(PtNDManager manager, long handle, ByteBuffer data) {
        super(handle);
        this.manager = manager;
        this.ptNDArrayEx = new PtNDArrayEx(this);
        manager.attachInternal(getUid(), this);
        dataRef = data;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return name;
    }

    /** {@inheritDoc} */
    @Override
    public void setName(String name) {
        this.name = name;
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        if (dataType == null) {
            dataType = JniUtils.getDataType(this);
        }
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        if (device == null) {
            device = JniUtils.getDevice(this);
        }
        return device;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape == null) {
            shape = JniUtils.getShape(this);
        }
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        if (sparseFormat == null) {
            sparseFormat = JniUtils.getSparseFormat(this);
        }
        return sparseFormat;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray toDevice(Device device, boolean copy) {
        if (device.equals(getDevice()) && !copy) {
            return this;
        }
        return JniUtils.to(this, getDataType(), device);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray toType(DataType dataType, boolean copy) {
        if (dataType.equals(getDataType()) && !copy) {
            return this;
        }
        return JniUtils.to(this, dataType, getDevice());
    }

    /** {@inheritDoc} */
    @Override
    public void setRequiresGradient(boolean requiresGrad) {
        JniUtils.attachGradient(this, requiresGrad);
        hasGradient = requiresGrad;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray getGradient() {
        if (!hasGradient()) {
            throw new IllegalStateException(
                    "No gradient attached to this NDArray, please call array.setRequiresGradient()"
                            + " on your NDArray or block.setInitializer() on your Block");
        }
        PtNDArray res = JniUtils.getGradient(this);
        // If you call getGradient() before you run the backward,
        // you will get nothing in PyTorch engine.
        // To align with MXNet's behavior, we will create a zeros NDArray.
        // TODO should we access the grad NDArray after we close the parameter NDArray?
        if (res == null) {
            res = (PtNDArray) manager.zeros(getShape());
        }
        return res;
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasGradient() {
        if (hasGradient == null) {
            hasGradient = JniUtils.requiresGrad(this);
        }
        return hasGradient;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stopGradient() {
        return JniUtils.detachGradient(this);
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return JniUtils.getByteBuffer(this);
    }

    /** {@inheritDoc} */
    @Override
    public String[] toStringArray(Charset charset) {
        throw new UnsupportedOperationException("String NDArray is not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        int size = Math.toIntExact(size());
        BaseNDManager.validateBufferSize(data, getDataType(), size);
        // TODO how do we handle the exception happened in the middle
        dataRef = null;
        if (data.isDirect() && data instanceof ByteBuffer) {
            // If NDArray is on the GPU, it is native code responsibility to control the data life
            // cycle
            if (!getDevice().isGpu()) {
                dataRef = (ByteBuffer) data;
            }
            JniUtils.set(this, (ByteBuffer) data);
            return;
        }
        // int8, uint8, boolean use ByteBuffer, so need to explicitly input DataType
        DataType inputType = DataType.fromBuffer(data);
        ByteBuffer buf = manager.allocateDirect(size * inputType.getNumOfBytes());
        BaseNDManager.copyBuffer(data, buf);

        // If NDArray is on the GPU, it is native code responsibility to control the data life cycle
        if (!getDevice().isGpu()) {
            dataRef = buf;
        }
        JniUtils.set(this, buf);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(long... indices) {
        return JniUtils.getItem(this, indices);
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray array) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public void attach(NDManager manager) {
        detach();
        this.manager = (PtNDManager) manager;
        manager.attachInternal(getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void tempAttach(NDManager manager) {
        detach();
        NDManager original = this.manager;
        this.manager = (PtNDManager) manager;
        manager.tempAttachInternal(original, getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = PtNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray duplicate() {
        return JniUtils.clone(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray booleanMask(NDArray index, int axis) {
        Shape indexShape = index.getShape();
        if (indexShape.equals(getShape())) {
            // Result is flattened since shape is undetermined
            return JniUtils.booleanMask(this, manager.from(index));
        } else if (indexShape.equals(getShape().slice(axis))) {
            // index will be broadcasted by default
            try (PtNDArray flattedResult = JniUtils.booleanMask(this, manager.from(index))) {
                // Shape recovery
                Shape remainder = getShape().slice(0, axis);
                long selectedSize = flattedResult.getShape().size() / remainder.size();
                return flattedResult.reshape(remainder.addAll(new Shape(selectedSize)));
            }
        } else {
            throw new UnsupportedOperationException(
                    "Not supported for shape not broadcastable "
                            + indexShape
                            + " vs "
                            + getShape());
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength, float value) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sequenceMask(NDArray sequenceLength) {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(Number number) {
        return JniUtils.contentEqual(this, (PtNDArray) manager.create(number));
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(NDArray other) {
        if (other == null || (!shapeEquals(other))) {
            return false;
        }
        if (getDataType() != other.getDataType()) {
            return false;
        }
        return JniUtils.contentEqual(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray eq(Number n) {
        try (NDArray number = manager.create(n)) {
            return eq(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray eq(NDArray other) {
        return JniUtils.eq(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray neq(Number n) {
        try (NDArray number = manager.create(n)) {
            return neq(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray neq(NDArray other) {
        return JniUtils.neq(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray gt(Number n) {
        try (NDArray number = manager.create(n)) {
            return gt(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray gt(NDArray other) {
        return JniUtils.gt(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray gte(Number n) {
        try (NDArray number = manager.create(n)) {
            return gte(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray gte(NDArray other) {
        return JniUtils.gte(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray lt(Number n) {
        try (NDArray number = manager.create(n)) {
            return lt(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray lt(NDArray other) {
        return JniUtils.lt(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray lte(Number n) {
        try (NDArray number = manager.create(n)) {
            return lte(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray lte(NDArray other) {
        return JniUtils.lte(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray add(Number n) {
        try (NDArray number = manager.create(n)) {
            return add(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray add(NDArray other) {
        return JniUtils.add(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sub(Number n) {
        try (NDArray number = manager.create(n)) {
            return sub(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sub(NDArray other) {
        return JniUtils.sub(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray mul(Number n) {
        try (NDArray number = manager.create(n)) {
            return mul(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray mul(NDArray other) {
        return JniUtils.mul(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray div(Number n) {
        try (NDArray number = manager.create(n)) {
            return div(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray div(NDArray other) {
        return JniUtils.div(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray mod(Number n) {
        try (NDArray number = manager.create(n)) {
            return mod(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray mod(NDArray other) {
        return JniUtils.remainder(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray pow(Number n) {
        try (NDArray number = manager.create(n)) {
            return pow(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray pow(NDArray other) {
        return JniUtils.pow(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray addi(Number n) {
        try (NDArray number = manager.create(n)) {
            return addi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray addi(NDArray other) {
        JniUtils.addi(this, manager.from(other));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray subi(Number n) {
        try (NDArray number = manager.create(n)) {
            return subi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray subi(NDArray other) {
        JniUtils.subi(this, manager.from(other));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray muli(Number n) {
        try (NDArray number = manager.create(n)) {
            return muli(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray muli(NDArray other) {
        JniUtils.muli(this, manager.from(other));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray divi(Number n) {
        try (NDArray number = manager.create(n)) {
            return divi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray divi(NDArray other) {
        JniUtils.divi(this, manager.from(other));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray modi(Number n) {
        try (NDArray number = manager.create(n)) {
            return modi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray modi(NDArray other) {
        JniUtils.remainderi(this, manager.from(other));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray powi(Number n) {
        try (NDArray number = manager.create(n)) {
            return powi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray powi(NDArray other) {
        JniUtils.powi(this, manager.from(other));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sign() {
        return JniUtils.sign(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray signi() {
        JniUtils.signi(this);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray maximum(Number n) {
        try (NDArray number = manager.create(n)) {
            return maximum(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray maximum(NDArray other) {
        return JniUtils.max(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray minimum(Number n) {
        try (NDArray number = manager.create(n)) {
            return minimum(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray minimum(NDArray other) {
        return JniUtils.min(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray all() {
        try (PtNDArray bool = toType(DataType.BOOLEAN, true)) {
            return JniUtils.all(bool);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray any() {
        try (PtNDArray bool = toType(DataType.BOOLEAN, true)) {
            return JniUtils.any(bool);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray none() {
        try (PtNDArray bool = toType(DataType.BOOLEAN, true)) {
            return JniUtils.none(bool);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray neg() {
        return JniUtils.neg(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray negi() {
        JniUtils.negi(this);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray abs() {
        return JniUtils.abs(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray square() {
        return JniUtils.square(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sqrt() {
        return JniUtils.sqrt(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray cbrt() {
        return JniUtils.pow(this, (PtNDArray) manager.create(1.0 / 3));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray floor() {
        return JniUtils.floor(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray ceil() {
        return JniUtils.ceil(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray round() {
        return JniUtils.round(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray trunc() {
        return JniUtils.trunc(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray exp() {
        return JniUtils.exp(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray log() {
        return JniUtils.log(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray log10() {
        return JniUtils.log10(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray log2() {
        return JniUtils.log2(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sin() {
        return JniUtils.sin(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray cos() {
        return JniUtils.cos(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray tan() {
        return JniUtils.tan(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray asin() {
        return JniUtils.asin(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray acos() {
        return JniUtils.acos(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray atan() {
        return JniUtils.atan(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sinh() {
        return JniUtils.sinh(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray cosh() {
        return JniUtils.cosh(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray tanh() {
        return JniUtils.tanh(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray asinh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray acosh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray atanh() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray toDegrees() {
        return mul(180.0).div(Math.PI);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray toRadians() {
        return mul(Math.PI).div(180.0);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray max() {
        return JniUtils.max(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray max(int[] axes, boolean keepDims) {
        if (axes.length > 1) {
            // TODO fix this
            throw new UnsupportedOperationException("Only 1 axis is support!");
        }
        return JniUtils.max(this, axes[0], keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray min() {
        return JniUtils.min(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray min(int[] axes, boolean keepDims) {
        if (axes.length > 1) {
            // TODO fix this
            throw new UnsupportedOperationException("Only 1 axis is support!");
        }
        return JniUtils.min(this, axes[0], keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sum() {
        return JniUtils.sum(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sum(int[] axes, boolean keepDims) {
        return JniUtils.sum(this, Arrays.stream(axes).mapToLong(i -> i).toArray(), keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray prod() {
        return JniUtils.prod(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray prod(int[] axes, boolean keepDims) {
        if (axes.length > 1) {
            throw new UnsupportedOperationException("Only 1 axis is support!");
        }
        return JniUtils.prod(this, axes[0], keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray mean() {
        return JniUtils.mean(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray mean(int[] axes, boolean keepDims) {
        if (axes.length > 1) {
            // TODO fix this
            throw new UnsupportedOperationException("Only 1 axis is support!");
        }
        return JniUtils.mean(this, axes[0], keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rotate90(int times, int[] axes) {
        if (axes.length != 2) {
            throw new IllegalArgumentException("Axes must be 2");
        }
        return JniUtils.rot90(this, times, axes);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray trace(int offset, int axis1, int axis2) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long sections, int axis) {
        long size = getShape().get(axis) / sections;
        return JniUtils.split(this, size, axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long[] indices, int axis) {
        if (indices.length == 0) {
            return new NDList(this);
        }
        List<Long> ptIndex = new ArrayList<>();
        ptIndex.add(indices[0]);
        for (int i = 1; i < indices.length; i++) {
            ptIndex.add(indices[i] - indices[i - 1]);
        }
        ptIndex.add(size(axis) - indices[indices.length - 1]);
        return JniUtils.split(this, ptIndex.stream().mapToLong(i -> i).toArray(), axis);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray flatten() {
        return JniUtils.flatten(this, 0, -1);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray reshape(Shape shape) {
        return JniUtils.reshape(this, shape.getShape());
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray expandDims(int axis) {
        return JniUtils.unsqueeze(this, axis);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray squeeze() {
        return JniUtils.squeeze(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray squeeze(int axis) {
        return JniUtils.squeeze(this, axis);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray squeeze(int[] axes) {
        if (isScalar()) {
            if (axes.length > 1 || axes[0] != 0) {
                throw new IllegalArgumentException(
                        "axis " + axes[0] + "is out of bounds for array of dimension 0");
            }
            return (PtNDArray) duplicate();
        }
        long[] shapeArr = getShape().getShape();
        List<Long> newShape = new ArrayList<>();
        Set<Integer> set =
                IntStream.of(axes).boxed().collect(Collectors.toCollection(HashSet::new));
        // check input
        for (int axis : axes) {
            if (shapeArr[axis] != 1) {
                throw new IllegalArgumentException(
                        "cannot select an axis to squeeze out which has size not equal to one");
            }
        }
        for (int i = 0; i < shapeArr.length; i++) {
            if (!set.contains(i)) {
                newShape.add(shapeArr[i]);
            }
        }
        return (PtNDArray) reshape(newShape.stream().mapToLong(i -> i).toArray());
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray logicalAnd(NDArray other) {
        return JniUtils.logicalAnd(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray logicalOr(NDArray other) {
        return JniUtils.logicalOr(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray logicalXor(NDArray other) {
        return JniUtils.logicalXor(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray logicalNot() {
        return JniUtils.logicalNot(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray argSort(int axis, boolean ascending) {
        if (!ascending) {
            throw new UnsupportedOperationException("Only support ascending!");
        }
        return JniUtils.argSort(this, axis, false);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sort() {
        return sort(-1);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sort(int axis) {
        return JniUtils.sort(this, axis, false);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray softmax(int axis) {
        return JniUtils.softmax(this, axis, getDataType());
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray logSoftmax(int axis) {
        return JniUtils.logSoftmax(this, axis, getDataType());
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray cumSum() {
        // TODO: change default behavior on cumSum
        if (isScalar()) {
            return (PtNDArray) reshape(1);
        }
        if (isEmpty()) {
            return (PtNDArray) reshape(0);
        }
        return cumSum(0);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray cumSum(int axis) {
        return JniUtils.cumSum(this, axis);
    }

    /** {@inheritDoc} */
    @Override
    public void intern(NDArray replaced) {
        PtNDArray arr = (PtNDArray) replaced;
        Long oldHandle = handle.getAndSet(arr.handle.getAndSet(null));
        JniUtils.deleteNDArray(oldHandle);
        // dereference old ndarray
        arr.close();
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray isInfinite() {
        return JniUtils.isInf(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray isNaN() {
        return JniUtils.isNaN(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray tile(long repeats) {
        // zero-dim
        if (isEmpty()) {
            return (PtNDArray) duplicate();
        }
        // scalar
        int dim = (isScalar()) ? 1 : getShape().dimension();
        long[] repeatsArray = new long[dim];
        Arrays.fill(repeatsArray, repeats);
        return tile(repeatsArray);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray tile(int axis, long repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray tile(long[] repeats) {
        return JniUtils.tile(this, repeats);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray tile(Shape desiredShape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray repeat(long repeats) {
        // zero-dim
        if (isEmpty()) {
            return (PtNDArray) duplicate();
        }
        // scalar
        int dim = (isScalar()) ? 1 : getShape().dimension();
        long[] repeatsArray = new long[dim];
        Arrays.fill(repeatsArray, repeats);
        return repeat(repeatsArray);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray repeat(int axis, long repeats) {
        return JniUtils.repeat(this, repeats, axis);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray repeat(long[] repeats) {
        PtNDArray result = this;
        for (int dim = 0; dim < repeats.length; dim++) {
            PtNDArray temp = result;
            result = JniUtils.repeat(result, repeats[dim], dim);
            if (temp != this) {
                temp.close();
            }
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray repeat(Shape desiredShape) {
        return repeat(repeatsToMatchShape(desiredShape));
    }

    private long[] repeatsToMatchShape(Shape desiredShape) {
        Shape curShape = getShape();
        int dimension = curShape.dimension();
        if (desiredShape.dimension() > dimension) {
            throw new IllegalArgumentException("The desired shape has too many dimensions");
        }
        if (desiredShape.dimension() < dimension) {
            int additionalDimensions = dimension - desiredShape.dimension();
            desiredShape = curShape.slice(0, additionalDimensions).addAll(desiredShape);
        }
        long[] repeats = new long[dimension];
        for (int i = 0; i < dimension; i++) {
            if (curShape.get(i) == 0 || desiredShape.get(i) % curShape.get(i) != 0) {
                throw new IllegalArgumentException(
                        "The desired shape is not a multiple of the original shape");
            }
            repeats[i] = Math.round(Math.ceil((double) desiredShape.get(i) / curShape.get(i)));
        }
        return repeats;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray dot(NDArray other) {
        int selfDim = this.getShape().dimension();
        int otherDim = other.getShape().dimension();
        if (selfDim != otherDim || selfDim > 2) {
            throw new UnsupportedOperationException(
                    "Dimension mismatch or high dimensional dot operation is not supported. Please use .matMul instead.");
        }
        return JniUtils.dot(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray matMul(NDArray other) {
        if (isScalar() || other.isScalar()) {
            throw new IllegalArgumentException("scalar is not allowed for matMul()");
        }
        return JniUtils.matmul(this, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray clip(Number min, Number max) {
        return JniUtils.clip(this, min, max);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray swapAxes(int axis1, int axis2) {
        return JniUtils.transpose(this, axis1, axis2);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flip(int... axes) {
        return JniUtils.flip(this, Arrays.stream(axes).mapToLong(ele -> (long) ele).toArray());
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray transpose() {
        int dim = getShape().dimension();
        int[] reversedShape = IntStream.range(0, dim).map(i -> dim - i - 1).toArray();
        return transpose(reversedShape);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray transpose(int... axes) {
        if (isScalar() && axes.length > 0) {
            throw new IllegalArgumentException("axes don't match NDArray");
        }
        return JniUtils.permute(this, Arrays.stream(axes).mapToLong(i -> i).toArray());
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray broadcast(Shape shape) {
        return JniUtils.broadcast(this, shape);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray argMax() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMax of an empty NDArray");
        }
        if (isScalar()) {
            return (PtNDArray) manager.create(0L);
        }
        return JniUtils.argMax(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray argMax(int axis) {
        // TODO pytorch bug: https://github.com/pytorch/pytorch/issues/37084
        if (isScalar()) {
            return (PtNDArray) manager.create(0L);
        }
        return JniUtils.argMax(this, axis, false);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray argMin() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMin of an empty NDArray");
        }
        if (isScalar()) {
            return (PtNDArray) manager.create(0L);
        }
        return JniUtils.argMin(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray argMin(int axis) {
        // TODO pytorch bug: https://github.com/pytorch/pytorch/issues/37084
        if (isScalar()) {
            return (PtNDArray) manager.create(0L);
        }
        return JniUtils.argMin(this, axis, false);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray percentile(Number percentile) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray percentile(Number percentile, int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray median() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray median(int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray toDense() {
        if (!isSparse() && JniUtils.getLayout(this) != 2) {
            return (PtNDArray) duplicate();
        }
        return JniUtils.toDense(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray toSparse(SparseFormat fmt) {
        if (fmt == SparseFormat.DENSE) {
            throw new IllegalArgumentException("Default type is not allowed");
        }
        if (fmt != SparseFormat.COO) {
            throw new UnsupportedOperationException("Only COO sparse type supported for PyTorch");
        }
        if (fmt == getSparseFormat()) {
            return (PtNDArray) duplicate();
        }
        return JniUtils.toSparse(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray nonzero() {
        return JniUtils.nonZeros(this);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray erfinv() {
        return JniUtils.erfinv(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm(boolean keepDims) {
        return JniUtils.norm(this, 2, new int[] {}, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm(int order, int[] axes, boolean keepDims) {
        return JniUtils.norm(this, order, axes, keepDims);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray oneHot(int depth) {
        return JniUtils.oneHot(this, depth, DataType.FLOAT32);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray oneHot(int depth, DataType dataType) {
        return JniUtils.oneHot(this, depth, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray oneHot(int depth, float onValue, float offValue, DataType dataType) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray batchDot(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArrayEx getNDArrayInternal() {
        return ptNDArrayEx;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (isReleased()) {
            return "This array is already closed";
        }
        // index operator in toDebugString is not supported for MKLDNN & Sparse layout
        if (JniUtils.getLayout(this) != 0) {
            try (NDArray tmp = toDense()) {
                return tmp.toDebugString();
            }
        }
        return toDebugString();
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof PtNDArray) {
            return contentEquals((PtNDArray) obj);
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deleteNDArray(pointer);
        }
        manager.detachInternal(getUid());
        dataRef = null;
    }
}
