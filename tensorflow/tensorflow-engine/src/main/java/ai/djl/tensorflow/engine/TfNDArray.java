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
package ai.djl.tensorflow.engine;

import ai.djl.Device;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.util.NativeResource;
import ai.djl.util.Pair;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.UUID;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import org.tensorflow.Operand;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Tensor;
import org.tensorflow.types.family.TType;

public class TfNDArray extends NativeResource<Operand<?>> implements NDArray {

    private static final int MAX_SIZE = 100;
    private static final int MAX_DEPTH = 10;
    private static final int MAX_ROWS = 10;
    private static final int MAX_COLUMNS = 20;

    private String uid;
    private Shape shape;
    private TfNDManager manager;
    private String name = "";
    private TfNDArrayEx tfNDArrayEx;
    private DataType dataType;

    TfNDArray(TfNDManager manager, Tensor tensor) {
        super(
                manager.getEagerSession()
                        .opBuilder("Const", "Const_" + TfNDManager.nextNameAssignment())
                        .setAttr("dtype", tensor.dataType())
                        .setAttr("value", tensor)
                        .setDevice(getTfDevice(manager.getDevice()))
                        .build()
                        .output(0));
        this.manager = manager;
        uid = UUID.randomUUID().toString();
        manager.attachInternal(uid, this);
        // cache shape and data type information so we can close tensor later
        this.shape = new Shape(tensor.shape().asArray());
        this.dataType = TfDataType.fromProtoType(tensor.dataType());
        tfNDArrayEx = new TfNDArrayEx(this);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
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
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return manager.getDevice();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSparse() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDevice(Device device, boolean copy) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toType(DataType dataType, boolean copy) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Cast", "Cast");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.setAttr("DstT", TfDataType.toProtoType(dataType));
        Operand<?> output = opBuilder.build().output(0);
        // FIXME: !copy is not implemented yet!
        try (Tensor tensor = output.asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void setRequiresGradient(boolean requiresGrad) {}

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasGradient() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stopGradient() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public double[] toDoubleArray() {
        if (getDataType() != DataType.FLOAT64) {
            throw new IllegalStateException(
                    "DataType mismatch, Required double" + " Actual " + getDataType());
        }
        double[] result = new double[(int) getShape().size()];
        try (Tensor tensor = getHandle().asTensor()) {
            tensor.asRawTensor().data().asDoubles().read(result);
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public float[] toFloatArray() {
        if (getDataType() != DataType.FLOAT32) {
            throw new IllegalStateException(
                    "DataType mismatch, Required float, Actual " + getDataType());
        }
        float[] result = new float[(int) getShape().size()];
        try (Tensor tensor = getHandle().asTensor()) {
            tensor.asRawTensor().data().asFloats().read(result);
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public int[] toIntArray() {
        if (getDataType() != DataType.INT32) {
            throw new IllegalStateException(
                    "DataType mismatch, Required int" + " Actual " + getDataType());
        }
        int[] result = new int[(int) getShape().size()];
        try (Tensor tensor = getHandle().asTensor().asRawTensor()) {
            tensor.asRawTensor().data().asInts().read(result);
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public long[] toLongArray() {
        if (getDataType() != DataType.INT64) {
            throw new IllegalStateException(
                    "DataType mismatch, Required long" + " Actual " + getDataType());
        }
        long[] result = new long[(int) getShape().size()];
        try (Tensor tensor = getHandle().asTensor().asRawTensor()) {
            tensor.asRawTensor().data().asLongs().read(result);
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public boolean[] toBooleanArray() {
        if (getDataType() != DataType.BOOLEAN) {
            throw new IllegalStateException(
                    "DataType mismatch, Required boolean" + " Actual " + getDataType());
        }
        boolean[] result = new boolean[(int) getShape().size()];
        try (Tensor tensor = getHandle().asTensor().asRawTensor()) {
            tensor.asRawTensor().data().asBooleans().read(result);
        }
        return result;
    }

    @Override
    public String[] toStringArray() {
        // TODO: Parse String Array from bytes[]
        throw new UnsupportedOperationException(
                "TensorFlow does not supporting printing String NDArray");
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        Shape sh = getShape();
        DataType dType = getDataType();
        long product = sh.size();
        long len = dType.getNumOfBytes() * product;
        byte[] buf = new byte[Math.toIntExact(len)];
        try (Tensor tensor = getHandle().asTensor().asRawTensor()) {
            tensor.asRawTensor().data().read(buf);
        }
        return ByteBuffer.wrap(buf);
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }

    /** {@inheritDoc} */
    @Override
    public void attach(NDManager manager) {
        detach();
        this.manager = (TfNDManager) manager;
        manager.attachInternal(uid, this);
    }

    /** {@inheritDoc} */
    @Override
    public void tempAttach(NDManager manager) {
        detach();
        NDManager original = this.manager;
        this.manager = (TfNDManager) manager;
        manager.tempAttachInternal(original, uid, this);
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = TfNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray ndArray) {
        if (!(ndArray instanceof TfNDArray)) {
            throw new IllegalArgumentException("Only TfNDArray is supported.");
        }
        Shape inShape = getShape();
        Shape destShape = ndArray.getShape();
        if (!Arrays.equals(inShape.getShape(), destShape.getShape())) {
            throw new IllegalArgumentException(
                    "shape are diff. Required: " + destShape + ", Actual " + inShape);
        }
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("DeepCopy", "DeepCopy");
        opBuilder.addInput(getHandle().asOutput());
        Operand<? extends TType> operand = opBuilder.build().output(0);
        ((TfNDArray) ndArray).handle.set(operand);
        ((TfNDArray) ndArray).dataType = getDataType();
        ((TfNDArray) ndArray).shape =
                new Shape(getShape().stream().mapToLong(Pair::getKey).toArray());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray booleanMask(NDArray index, int axis) {
        // FIXME re-implement
        throw new UnsupportedOperationException("Not implemented yet");
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
    public NDArray zerosLike() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("ZerosLike", "ZerosLike");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("OnesLike", "OnesLike");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(Number number) {
        if (number == null) {
            return false;
        }
        try (NDArray result = eq(number)) {
            return result.all().getBoolean();
        }
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
        TfNDArray eq = (TfNDArray) eq(other);
        return eq.all().toBooleanArray()[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return eq(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Equal", "Equal");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return neq(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("NotEqual", "NotEqual");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return gt(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Greater", "Greater");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return gte(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        OperationBuilder opBuilder =
                manager.getEagerSession().opBuilder("GreaterEqual", "GreaterEqual");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return lt(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Less", "Less");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return lte(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("LessEqual", "LessEqual");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray all() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("All", "ReduceAll");
        opBuilder.addInput(((TfNDArray) toType(DataType.BOOLEAN, false)).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.arange(getRank())).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray any() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Any", "ReduceAny");
        opBuilder.addInput(((TfNDArray) toType(DataType.BOOLEAN, false)).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.arange(getRank())).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray erfinv() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Erfinv", "erfinv");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm(boolean keepDims) {
        // We have to flatten first to be able to simulate "numpy.linalg.norm" whenever axis isn't
        // specified
        if (dataType == DataType.FLOAT64) {
            throw new UnsupportedOperationException("float64 is not supported");
        }
        NDArray flatten = flatten();
        OperationBuilder opBuilder =
                manager.getEagerSession().opBuilder("EuclideanNorm", "EuclideanNorm");
        opBuilder.addInput(((TfNDArray) flatten).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(0)).getHandle().asOutput());
        opBuilder.setAttr("keep_dims", keepDims);
        // close the temp NDArray
        flatten.close();
        if (!keepDims) {
            try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
                return new TfNDArray(manager, tensor);
            }
        }
        long[] shapes = LongStream.generate(() -> 1).limit(shape.dimension()).toArray();
        try (Tensor tensor = opBuilder.build().output(0).asTensor();
                NDArray temp = new TfNDArray(manager, tensor)) {
            return temp.reshape(shapes);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm(int ord, int[] axes, boolean keepDims) {
        if (ord != 2) {
            throw new UnsupportedOperationException("Only ord=2 is supported");
        }
        OperationBuilder opBuilder =
                manager.getEagerSession().opBuilder("EuclideanNorm", "EuclideanNorm");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(axes)).getHandle().asOutput());
        opBuilder.setAttr("keep_dims", keepDims);
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public NDArray oneHot(int depth, float onValue, float offValue, DataType dataType) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("OneHot", "OneHot");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(depth)).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(onValue)).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(offValue)).getHandle().asOutput());
        opBuilder.setAttr("axis", -1);
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor).toType(dataType, false);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return add(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Add", "Add");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return sub(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Sub", "Sub");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return mul(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Mul", "Mul");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return div(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Div", "Div");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return mod(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Mod", "Mod");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return pow(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(NDArray other) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Pow", "Pow");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return maximum(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(NDArray other) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Maximum", "Maximum");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return minimum(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(NDArray other) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Minimum", "inimum");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) other).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return addi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        return inPlaceHelper(add(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return subi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        return inPlaceHelper(sub(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return muli(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other) {
        return inPlaceHelper(mul(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return divi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        return inPlaceHelper(div(other), this);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    NDArray inPlaceHelper(NDArray source, NDArray destination) {
        if (getShape().isScalar()) {
            throw new UnsupportedOperationException(
                    "TensorFlow engine does not support inplace operations on scalars yet");
        }
        // select all indices for inplace update
        NDArray indices = manager.arange((int) getShape().getShape()[0]);
        OperationBuilder opBuilder =
                manager.getEagerSession().opBuilder("InplaceUpdate", "InplaceUpdate");
        opBuilder.addInput(((TfNDArray) destination).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) indices).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) source).getHandle().asOutput());
        ((TfNDArray) destination).setOperand(opBuilder.build().output(0));
        return destination;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toSparse(SparseFormat fmt) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return modi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        return inPlaceHelper(mod(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(Number n) {
        try (NDArray number = manager.create(n).toType(getDataType(), false)) {
            return powi(number);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(NDArray other) {
        return inPlaceHelper(pow(other), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sign() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Sign", "Sign");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray signi() {
        return inPlaceHelper(sign(), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Neg", "Neg");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        return inPlaceHelper(neg(), this);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Abs", "Abs");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Square", "Square");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sqrt() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Sqrt", "Sqrt");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Pow", "Pow");
        opBuilder.addInput(getHandle().asOutput());
        if (getDataType().equals(DataType.FLOAT64)) {
            opBuilder.addInput(((TfNDArray) manager.create(1.0 / 3)).getHandle().asOutput());
        } else {
            opBuilder.addInput(((TfNDArray) manager.create(1f / 3)).getHandle().asOutput());
        }
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Floor", "Floor");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Ceil", "Ceil");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Round", "Round");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Exp", "Exp");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Log", "Log");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        return log().div(Math.log(10));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        return log().div(Math.log(2));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Sin", "Sin");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Cos", "Cos");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Tan", "Tan");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Asin", "Asin");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Acos", "Acos");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Atan", "Atan");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Sinh", "Sinh");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Cosh", "Cosh");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Tanh", "Tanh");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Asinh", "Asinh");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Acosh", "Acosh");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Atanh", "Atanh");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        return mul(180).div(Math.PI);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        return mul(Math.PI).div(180);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Max", "Max");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.arange(0, getRank(), 1)).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Max", "Max");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(axes)).getHandle().asOutput());
        opBuilder.setAttr("keep_dims", keepDims);
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Min", "Min");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.arange(0, getRank(), 1)).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Min", "Min");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(axes)).getHandle().asOutput());
        opBuilder.setAttr("keep_dims", keepDims);
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum() {
        // sum on all axis
        NDArray array = this;
        // tf can't sum boolean values
        if (getDataType() == DataType.BOOLEAN) {
            array = array.toType(DataType.INT64, false);
        }
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Sum", "Sum");
        opBuilder.addInput(((TfNDArray) array).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.arange(0, getRank(), 1)).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Sum", "Sum");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(axes)).getHandle().asOutput());
        opBuilder.setAttr("keep_dims", keepDims);
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Prod", "Prod");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.arange(0, getRank())).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Prod", "Prod");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput((((TfNDArray) manager.create(axes)).getHandle().asOutput()));
        opBuilder.setAttr("keep_dims", keepDims);
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Mean", "Mean");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput((((TfNDArray) manager.arange(0, getRank())).getHandle().asOutput()));
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Mean", "Mean");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput((((TfNDArray) manager.create(axes)).getHandle().asOutput()));
        opBuilder.setAttr("keep_dims", keepDims);
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rotate90(int times, int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(long[] indices, int axis) {
        // FIXME fix in next PR
        throw new UnsupportedOperationException("Not Implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        return reshape(new Shape(-1));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Reshape", "Reshape");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput((((TfNDArray) manager.create(shape.getShape())).getHandle().asOutput()));
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        OperationBuilder opBuilder =
                manager.getEagerSession().opBuilder("ExpandDims", "ExpandDims");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput((((TfNDArray) manager.create(axis)).getHandle().asOutput()));
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        if (isScalar()) {
            axes = new int[0];
        }
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Squeeze", "Squeeze");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.setAttr("squeeze_dims", Arrays.stream(axes).mapToLong(i -> i).toArray());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalAnd(NDArray n) {
        OperationBuilder opBuilder =
                manager.getEagerSession().opBuilder("LogicalAnd", "LogicalAnd");
        opBuilder.addInput(((TfNDArray) toType(DataType.BOOLEAN, false)).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) n.toType(DataType.BOOLEAN, false)).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray n) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("LogicalOr", "LogicalOr");
        opBuilder.addInput(((TfNDArray) toType(DataType.BOOLEAN, false)).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) n.toType(DataType.BOOLEAN, false)).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray n) {
        return logicalOr(n).logicalAnd(logicalAnd(n).logicalNot());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        OperationBuilder opBuilder =
                manager.getEagerSession().opBuilder("LogicalNot", "LogicalNot");
        opBuilder.addInput(((TfNDArray) toType(DataType.BOOLEAN, false)).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argSort(int axis, boolean ascending) {
        // FIXME fix in next PR
        throw new UnsupportedOperationException("Not Implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        // FIXME fix in next PR
        throw new UnsupportedOperationException("Not Implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        // FIXME fix in next PR
        throw new UnsupportedOperationException("Not Implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int axis) {
        // FIXME fix in next PR
        throw new UnsupportedOperationException("Not Implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logSoftmax(int axis) {
        // FIXME fix in next PR
        throw new UnsupportedOperationException("Not Implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum(int axis) {
        // just expand dim for scalar
        if (isScalar()) {
            return expandDims(0);
        }
        // return 0 shape if any of the dim is 0
        if (Arrays.stream(getShape().getShape()).anyMatch(dim -> dim == 0L)) {
            return manager.create(new Shape(0));
        }
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Cumsum", "Cumsum");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput((((TfNDArray) manager.create(axis)).getHandle().asOutput()));
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum() {
        return cumSum(0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("IsInf", "IsInf");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("IsNan", "IsNan");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long repeats) {
        long[] multiples = new long[getShape().dimension()];
        Arrays.fill(multiples, repeats);
        return tile(multiples);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, long repeats) {
        long[] multiples = new long[getShape().dimension()];
        Arrays.fill(multiples, 1);
        multiples[axis] = repeats;
        return tile(multiples);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long[] repeats) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Tile", "Tile");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(repeats)).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(Shape desiredShape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int axis, long repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long[] repeats) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(Shape desiredShape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dot(NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray matMul(NDArray other) {
        // FIXME fix in next PR
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(Number min, Number max) {
        OperationBuilder opBuilder =
                manager.getEagerSession().opBuilder("ClipByValue", "ClipByValueClipByValue");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(min.floatValue())).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(max.floatValue())).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flip(int... axes) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("ReverseV2", "Reverse");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(axes)).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        int dim = getShape().dimension();
        int[] reversedShape = IntStream.range(0, dim).map(i -> dim - i - 1).toArray();
        return transpose(reversedShape);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int... dimensions) {
        if (Arrays.stream(dimensions).anyMatch(d -> d < 0)) {
            throw new UnsupportedOperationException(
                    "Passing -1 for broadcasting the dimension is not currently supported");
        }
        if (!Arrays.equals(
                Arrays.stream(dimensions).sorted().toArray(),
                IntStream.range(0, getShape().dimension()).toArray())) {
            throw new IllegalArgumentException(
                    "You must include each of the dimensions from 0 until "
                            + getShape().dimension());
        }
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("Transpose", "Transpose");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(dimensions)).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        OperationBuilder opBuilder =
                manager.getEagerSession().opBuilder("BroadcastTo", "BroadcastTo");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(shape.getShape())).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMax of an empty NDArray");
        }
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("ArgMax", "ArgMax");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("ArgMax", "ArgMax");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(axis)).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin() {
        if (isEmpty()) {
            throw new IllegalArgumentException("attempt to get argMin of an empty NDArray");
        }
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("ArgMin", "ArgMin");
        opBuilder.addInput(getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin(int axis) {
        OperationBuilder opBuilder = manager.getEagerSession().opBuilder("ArgMin", "ArgMin");
        opBuilder.addInput(getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) manager.create(axis)).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(manager, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int[] dimension) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median(int[] axes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDense() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray nonzero() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayEx getNDArrayInternal() {
        return tfNDArrayEx;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof TfNDArray) {
            return contentEquals((TfNDArray) obj);
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
    public String toString() {
        if (isReleased()) {
            return "This array is already closed";
        }

        return toDebugString(MAX_SIZE, MAX_DEPTH, MAX_ROWS, MAX_COLUMNS);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Operand<?> operand = handle.getAndSet(null);
        if (operand != null) {
            operand.asTensor().close();
            manager.detachInternal(getUid());
            manager = null;
        }
        tfNDArrayEx = null;
    }

    private static String getTfDevice(Device device) {
        if (device.getDeviceType().equals(Device.Type.CPU)) {
            return "/device:CPU:0";
        } else if (device.getDeviceType().equals(Device.Type.GPU)) {
            return "/device:GPU:" + device.getDeviceId();
        } else {
            throw new EngineException(
                    "Unknown device type to TensorFlow Engine: " + device.toString());
        }
    }

    public Tensor getTensor() {
        return getHandle().asTensor();
    }

    void setOperand(Operand<?> operand) {
        handle.set(operand);
        uid = handle.toString();
    }

    int getRank() {
        return getShape().dimension();
    }

    public static org.tensorflow.ndarray.Shape toTfShape(Shape shape) {
        return org.tensorflow.ndarray.Shape.of(shape.getShape());
    }
}
