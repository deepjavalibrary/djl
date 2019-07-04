package org.tensorflow.engine;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;
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

public class TfNDArray implements NDArray {

    private Tensor<?> tensor;
    private Output<?> out;
    private Shape shape;
    private TfNDFactory factory;

    TfNDArray(NDFactory factory, Tensor<?> tensor) {
        this.factory = (TfNDFactory) factory;
        this.factory.attach(this);
        this.tensor = tensor;
    }

    TfNDArray(NDFactory factory, Output<?> out) {
        this.factory = (TfNDFactory) factory;
        this.factory.attach(this);
        this.out = out;
    }

    public TfNDArray(NDFactory factory, Shape shape, FloatBuffer data) {
        this.factory = (TfNDFactory) factory;
        this.factory.attach(this);
        tensor = Tensor.create(shape.getShapeLong(), data);
        this.shape = shape;
    }

    TfNDArray(NDFactory factory, Shape shape, ByteBuffer data) {
        this.factory = (TfNDFactory) factory;
        tensor = Tensor.create(UInt8.class, shape.getShapeLong(), data);
        this.shape = shape;
    }

    Output<?> getOutput() {
        if (out == null) {
            Operation op =
                    factory.getGraph()
                            .opBuilder("Const", "Const_" + TfNDFactory.nextNameAssignment())
                            .setAttr("dtype", tensor.dataType())
                            .setAttr("value", tensor)
                            .build();
            out = op.output(0);
        }
        return out;
    }

    private void runToTensor() {
        if (tensor == null) {
            tensor = factory.getSession().runner().fetch(out.op().name()).run().get(0);
        }
    }

    /** {@inheritDoc} */
    @Override
    public byte[] getEncoded() {
        return new byte[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDFactory getFactory() {
        return factory;
    }

    public org.tensorflow.DataType getTfDataType() {
        if (tensor != null) {
            return tensor.dataType();
        } else {
            return out.dataType();
        }
    }

    /** /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return DataTypeMapper.getJoule(getTfDataType());
    }

    /** {@inheritDoc} */
    @Override
    public Context getContext() {
        return factory.getContext();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape == null) {
            runToTensor();
            shape = new Shape(Arrays.stream(tensor.shape()).mapToInt(Math::toIntExact).toArray());
        }
        return shape;
    }

    public Tensor<?> getTensor() {
        return tensor;
    }

    /** {@inheritDoc} */
    @Override
    public Layout getLayout() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc getDataDescriptor() {
        return new DataDesc(getShape(), getDataType(), null, getLayout(), getContext());
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }

    /** {@inheritDoc} */
    @Override
    public void set(float[] data) {}

    /** {@inheritDoc} */
    @Override
    public void set(int[] data) {}

    /** {@inheritDoc} */
    @Override
    public void set(double[] data) {}

    /** {@inheritDoc} */
    @Override
    public void set(long[] data) {}

    /** {@inheritDoc} */
    @Override
    public void set(byte[] data) {}

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDIndex index) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getElement(NDIndex index) throws IllegalArgumentException {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public long getLong(NDIndex index) throws IllegalArgumentException {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public double getDouble(NDIndex index) throws IllegalArgumentException {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public float getFloat(NDIndex index) throws IllegalArgumentException {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray set(NDIndex index, NDArray value) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray set(NDIndex index, Number value) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray setElement(NDIndex index, Number value) throws IllegalArgumentException {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray seti(NDIndex index, NDArray value) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray seti(NDIndex index, Number value) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray setElementi(NDIndex index, Number value) throws IllegalArgumentException {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray array) {}

    /** {@inheritDoc} */
    @Override
    public NDArray asInContext(Context ctx, boolean copy) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asType(DataType dtype, boolean copy) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void attachGrad() {}

    @Override
    public void attachGrad(GradReq gradReq, SparseFormat sparseFormat) {}

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void backward() {}

    /** {@inheritDoc} */
    @Override
    public void backward(boolean retainGraph, boolean isTraining) {}

    /** {@inheritDoc} */
    @Override
    public void backward(NDArray outGrad, boolean retainGraph, boolean isTraining) {}

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        return null;
    }

    @Override
    public NDArray argsort(int axis, boolean ascending) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes, double temperature) {
        Operation op =
                factory.getGraph()
                        .opBuilder("Softmax", "Softmax_" + TfNDFactory.nextNameAssignment())
                        .setAttr("T", getTfDataType())
                        .addInput(getOutput())
                        .build();

        return new TfNDArray(factory, op.output(0));
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int axis, boolean squeezeAxis) {
        TfNDArray axisOp = (TfNDArray) factory.create(axis);
        Operation op =
                factory.getGraph()
                        .opBuilder("Split", "Split_" + TfNDFactory.nextNameAssignment())
                        .setAttr("T", getTfDataType())
                        .setAttr("num_split", size(axis))
                        .addInput(axisOp.getOutput())
                        .addInput(getOutput())
                        .build();

        NDArray[] result =
                IntStream.range(0, op.numOutputs())
                        .mapToObj((int i) -> new TfNDArray(factory, op.output(i)))
                        .toArray(NDArray[]::new);
        return new NDList(result);
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int axis, int numOutputs) throws IllegalArgumentException {
        if (axis < 0 || axis > getShape().dimension()) {
            throw new IllegalArgumentException("Invalid axis value");
        }
        if (numOutputs < 0 || numOutputs > size(axis)) {
            throw new IllegalArgumentException("Invalid numOutputs");
        }
        TfNDArray axisOp = (TfNDArray) factory.create(axis);
        Operation op =
                factory.getGraph()
                        .opBuilder("Split", "Split_" + TfNDFactory.nextNameAssignment())
                        .setAttr("T", getTfDataType())
                        .setAttr("num_split", numOutputs)
                        .addInput(axisOp.getOutput())
                        .addInput(getOutput())
                        .build();

        NDArray[] result =
                IntStream.range(0, op.numOutputs())
                        .mapToObj((int i) -> new TfNDArray(factory, op.output(i)))
                        .toArray(NDArray[]::new);
        return new NDList(result);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike() {
        Operation op =
                factory.getGraph()
                        .opBuilder("ZerosLike", "ZerosLike_" + TfNDFactory.nextNameAssignment())
                        .setAttr("T", getTfDataType())
                        .addInput(getOutput())
                        .build();
        return new TfNDArray(factory, op.output(0));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        Operation op =
                factory.getGraph()
                        .opBuilder("OnesLike", "OnesLike_" + TfNDFactory.nextNameAssignment())
                        .setAttr("T", getTfDataType())
                        .addInput(getOutput())
                        .build();
        return new TfNDArray(factory, op.output(0));
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSparse() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsumi(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsumi() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsum(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsum() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eps(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eps(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(NDArray other) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(Number number) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        return null;
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
    public int nonzero() {
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

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        return null;
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
    public NDArray tile(int repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, int repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int[] repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(Shape desiredShape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int axis, int repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int[] repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(Shape desiredShape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mmul(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public double[] toDoubleArray() {
        if (getDataType() != DataType.FLOAT64) {
            throw new IllegalStateException(
                    "DataType mismatch, Required double" + " Actual " + getDataType());
        }
        runToTensor();
        DoubleBuffer db = DoubleBuffer.allocate(Math.toIntExact(size()));
        tensor.writeTo(db.duplicate());
        double[] ret = new double[Math.toIntExact(size())];
        db.get(ret);
        return ret;
    }

    /** {@inheritDoc} */
    @Override
    public float[] toFloatArray() {
        if (getDataType() != DataType.FLOAT32) {
            throw new IllegalStateException(
                    "DataType mismatch, Required float" + " Actual " + getDataType());
        }
        runToTensor();
        FloatBuffer fb = FloatBuffer.allocate(Math.toIntExact(size()));
        tensor.writeTo(fb.duplicate());
        float[] ret = new float[Math.toIntExact(size())];
        fb.get(ret);
        return ret;
    }

    /** {@inheritDoc} */
    @Override
    public int[] toIntArray() {
        return new int[0];
    }

    /** {@inheritDoc} */
    @Override
    public long[] toLongArray() {
        return new long[0];
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amax(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number amaxNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amin(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number aminNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number max() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number min() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number sum() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number prod() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number mean() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dup() {
        return null;
    }

    /** {@inheritDoc} */
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
    public boolean equalsWithEps(Object o, double eps) {
        return false;
    }

    @Override
    public boolean equalShapes(NDArray other) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public boolean none() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray like() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayEx getNDArrayInternal() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        return null;
    }

    /** {@inheritDoc} */
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

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (tensor != null) {
            tensor.close();
        }
        tensor = null;
    }
}
