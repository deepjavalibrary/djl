package org.tensorflow.engine;

import com.amazon.ai.Context;
import com.amazon.ai.ndarray.Matrix;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDFuncParams;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.GradReq;
import com.amazon.ai.ndarray.types.Layout;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.locks.Condition;
import java.util.stream.IntStream;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

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

    TfNDArray(NDFactory factory, Shape shape, FloatBuffer data) {
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
    public void encode(OutputStream os) throws IOException {}

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
        return new DataDesc(getShape(), getDataType(), null, getLayout(), getContext(), null);
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }

    /** {@inheritDoc} */
    @Override
    public void set(List<Float> data) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray at(int index) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray slice(int begin, int end) {
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

    /** {@inheritDoc} */
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
    public NDArray argsort(int axis, boolean ascending, NDFuncParams fparams) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes, Double temperature, NDFuncParams fparams) {
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
    public NDList split(int axis, boolean squeezeAxis, NDFuncParams fparams) {
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
    public NDList split(int axis, int numOutputs, NDFuncParams fparams)
            throws IllegalArgumentException {
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
    public NDArray zerosLike(NDFuncParams fparams) {
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
    public NDArray onesLike(NDFuncParams fparams) {
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
    public NDArray cumsumi(int dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumsum(int dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray assign(NDArray arr) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray assignIf(NDArray arr, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray replaceWhere(NDArray arr, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putScalar(long value, long... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putScalar(double value, long... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putScalar(float value, long... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putScalar(int value, long... dimension) {
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
    public NDArray rdiv(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(Number n) {
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
    public NDArray rdiv(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray match(NDArray comp, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray match(Number comp, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getWhere(NDArray comp, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getWhere(Number comp, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putWhere(NDArray comp, NDArray put, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putWhere(Number comp, NDArray put, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putWhereWithMask(NDArray mask, NDArray put) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putWhereWithMask(NDArray mask, Number put) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putWhere(Number comp, Number put, Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDArray indices) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(List<List<Integer>> indices) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray assign(Number value) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putSlice(int slice, NDArray put) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cond(Condition condition) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repmat(int... shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int dimension, long... repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getScalar(long i) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public double squaredDistance(NDArray other) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public double distance2(NDArray other) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public double distance1(NDArray other) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray put(List<List<Integer>> indices, NDArray element) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray put(NDArray indices, NDArray element) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray put(NDArray element, int... indices) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray put(int i, NDArray element) {
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
    public NDArray mmul(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mmuli(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mmuli(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray normmax(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number normmaxNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm2(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number norm2Number() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray norm1(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number norm1Number() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray std(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number stdNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray std(boolean biasCorrected, int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number stdNumber(boolean biasCorrected) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(NDArray result, int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amean(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number meanNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number ameanNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray var(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray var(boolean biasCorrected, int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number varNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amax(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number maxNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number amaxNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray amin(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number minNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number aminNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(boolean keepDims, int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(NDArray result, int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number sumNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number entropyNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number shannonEntropyNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number logEntropyNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray entropy(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray shannonEntropy(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logEntropy(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getScalar(int... indices) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getScalar(long... indices) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public long getLong(int... indices) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public long getLong(long... indices) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public double getDouble(int... indices) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public double getDouble(long... indices) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public float getFloat(int... indices) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public float getFloat(long... indices) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dup() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ravel() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ravel(char order) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray slice(long i, int dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray slice(long i) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(char order, long... newShape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(char order, int... newShape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(long... newShape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(int[] shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray swapAxes(int dimension, int with) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int... dimensions) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transposei(int... dimensions) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public long size() {
        if (tensor != null) {
            return tensor.numElements();
        } else {
            int dimensions = out.shape().numDimensions();
            return IntStream.range(0, dimensions)
                    .mapToLong((int i) -> out.shape().size(i))
                    .reduce(1, (long a, long b) -> a * b);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(long... shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Object element() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equalsWithEps(Object o, double eps) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equalShapes(NDArray other) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray remainder(NDArray denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray remainder(NDArray denominator, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray remainder(Number denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray remainder(Number denominator, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray remainderi(NDArray denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray remainderi(Number denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fmod(NDArray denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fmod(NDArray denominator, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fmod(Number denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fmod(Number denominator, NDArray result) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fmodi(NDArray denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray fmodi(Number denominator) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number percentileNumber(Number percentile) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Number medianNumber() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median(int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int... dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDense() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public int nonzero() {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isEmpty() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray castTo(DataType dataType) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Matrix asMatrix() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean all() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public boolean any() {
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
    public NDArray ulike() {
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
