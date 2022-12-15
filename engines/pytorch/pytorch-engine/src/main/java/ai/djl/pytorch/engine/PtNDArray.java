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

public interface PtNDArray extends NativeResource<Long>, NDArray {
    @Override
    PtNDManager getManager();

    @Override
    String getName();

    @Override
    void setName(String name);

    @Override
    DataType getDataType();

    @Override
    Device getDevice();

    @Override
    Shape getShape();

    @Override
    SparseFormat getSparseFormat();

    @Override
    PtNDArray toDevice(Device device, boolean copy);

    @Override
    PtNDArray toType(DataType dataType, boolean copy);

    @Override
    void setRequiresGradient(boolean requiresGrad);

    @Override
    PtNDArray getGradient();

    @Override
    boolean hasGradient();

    @Override
    NDArray stopGradient();

    @Override
    ByteBuffer toByteBuffer();

    @Override
    String[] toStringArray(Charset charset);

    @Override
    void set(Buffer buffer);

    @Override
    NDArray get(NDManager manager, long... indices);

    @Override
    NDArray gather(NDArray index, int axis);

    @Override
    NDArray gatherNd(NDArray index);

    @Override
    NDArray take(NDManager manager, NDArray index);

    @Override
    NDArray put(NDArray index, NDArray data);

    @Override
    void copyTo(NDArray array);

    @Override
    void attach(NDManager manager);

    @Override
    void returnResource(NDManager manager);

    @Override
    void tempAttach(NDManager manager);

    @Override
    void detach();

    @Override
    NDArray duplicate();

    @Override
    PtNDArray booleanMask(NDArray index, int axis);

    @Override
    NDArray sequenceMask(NDArray sequenceLength, float value);

    @Override
    NDArray sequenceMask(NDArray sequenceLength);

    @Override
    boolean contentEquals(Number number);

    @Override
    boolean contentEquals(NDArray other);

    @Override
    PtNDArray eq(Number n);

    @Override
    PtNDArray eq(NDArray other);

    @Override
    PtNDArray neq(Number n);

    @Override
    PtNDArray neq(NDArray other);

    @Override
    PtNDArray gt(Number n);

    @Override
    PtNDArray gt(NDArray other);

    @Override
    PtNDArray gte(Number n);

    @Override
    PtNDArray gte(NDArray other);

    @Override
    PtNDArray lt(Number n);

    @Override
    PtNDArray lt(NDArray other);

    @Override
    PtNDArray lte(Number n);

    @Override
    PtNDArray lte(NDArray other);

    @Override
    PtNDArray add(Number n);

    @Override
    PtNDArray add(NDArray other);

    @Override
    PtNDArray sub(Number n);

    @Override
    PtNDArray sub(NDArray other);

    @Override
    PtNDArray mul(Number n);

    @Override
    PtNDArray mul(NDArray other);

    @Override
    PtNDArray div(Number n);

    @Override
    PtNDArray div(NDArray other);

    @Override
    PtNDArray mod(Number n);

    @Override
    PtNDArray mod(NDArray other);

    @Override
    PtNDArray pow(Number n);

    @Override
    PtNDArray pow(NDArray other);

    @Override
    PtNDArray addi(Number n);

    @Override
    PtNDArray addi(NDArray other);

    @Override
    PtNDArray subi(Number n);

    @Override
    PtNDArray subi(NDArray other);

    @Override
    PtNDArray muli(Number n);

    @Override
    PtNDArray muli(NDArray other);

    @Override
    PtNDArray divi(Number n);

    @Override
    PtNDArray divi(NDArray other);

    @Override
    PtNDArray modi(Number n);

    @Override
    PtNDArray modi(NDArray other);

    @Override
    PtNDArray powi(Number n);

    @Override
    PtNDArray powi(NDArray other);

    @Override
    PtNDArray sign();

    @Override
    PtNDArray signi();

    @Override
    PtNDArray maximum(Number n);

    @Override
    PtNDArray maximum(NDArray other);

    @Override
    PtNDArray minimum(Number n);

    @Override
    PtNDArray minimum(NDArray other);

    @Override
    PtNDArray all();

    @Override
    PtNDArray any();

    @Override
    PtNDArray none();

    @Override
    PtNDArray neg();

    @Override
    PtNDArray negi();

    @Override
    PtNDArray abs();

    @Override
    PtNDArray square();

    @Override
    NDArray sqrt();

    @Override
    PtNDArray cbrt();

    @Override
    PtNDArray floor();

    @Override
    PtNDArray ceil();

    @Override
    PtNDArray round();

    @Override
    PtNDArray trunc();

    @Override
    PtNDArray exp();

    @Override
    NDArray gammaln();

    @Override
    PtNDArray log();

    @Override
    PtNDArray log10();

    @Override
    PtNDArray log2();

    @Override
    PtNDArray sin();

    @Override
    PtNDArray cos();

    @Override
    PtNDArray tan();

    @Override
    PtNDArray asin();

    @Override
    PtNDArray acos();

    @Override
    PtNDArray atan();

    @Override
    PtNDArray sinh();

    @Override
    PtNDArray cosh();

    @Override
    PtNDArray tanh();

    @Override
    PtNDArray asinh();

    @Override
    PtNDArray acosh();

    @Override
    PtNDArray atanh();

    @Override
    PtNDArray toDegrees();

    @Override
    PtNDArray toRadians();

    @Override
    PtNDArray max();

    @Override
    PtNDArray max(int[] axes, boolean keepDims);

    @Override
    PtNDArray min();

    @Override
    PtNDArray min(int[] axes, boolean keepDims);

    @Override
    PtNDArray sum();

    @Override
    PtNDArray sum(int[] axes, boolean keepDims);

    @Override
    NDArray cumProd(int axis);

    @Override
    NDArray cumProd(int axis, DataType dataType);

    @Override
    PtNDArray prod();

    @Override
    PtNDArray prod(int[] axes, boolean keepDims);

    @Override
    PtNDArray mean();

    @Override
    PtNDArray mean(int[] axes, boolean keepDims);

    @Override
    PtNDArray normalize(double p, long dim, double eps);

    @Override
    PtNDArray rotate90(int times, int[] axes);

    @Override
    PtNDArray trace(int offset, int axis1, int axis2);

    @Override
    NDList split(long sections, int axis);

    @Override
    NDList split(long[] indices, int axis);

    @Override
    PtNDArray flatten();

    @Override
    NDArray flatten(int startDim, int endDim);

    @Override
    PtNDArray reshape(Shape shape);

    @Override
    PtNDArray expandDims(int axis);

    @Override
    PtNDArray squeeze();

    @Override
    PtNDArray squeeze(int axis);

    @Override
    PtNDArray squeeze(int[] axes);

    @Override
    PtNDArray logicalAnd(NDArray other);

    @Override
    PtNDArray logicalOr(NDArray other);

    @Override
    PtNDArray logicalXor(NDArray other);

    @Override
    PtNDArray logicalNot();

    @Override
    PtNDArray argSort(int axis, boolean ascending);

    @Override
    PtNDArray sort();

    @Override
    PtNDArray sort(int axis);

    @Override
    PtNDArray softmax(int axis);

    @Override
    PtNDArray logSoftmax(int axis);

    @Override
    PtNDArray cumSum();

    @Override
    PtNDArray cumSum(int axis);

    @Override
    void intern(NDArray replaced);

    @Override
    PtNDArray isInfinite();

    @Override
    PtNDArray isNaN();

    @Override
    PtNDArray tile(long repeats);

    @Override
    PtNDArray tile(int axis, long repeats);

    @Override
    PtNDArray tile(long[] repeats);

    @Override
    PtNDArray tile(Shape desiredShape);

    @Override
    PtNDArray repeat(long repeats);

    @Override
    PtNDArray repeat(int axis, long repeats);

    @Override
    PtNDArray repeat(long[] repeats);

    @Override
    PtNDArray repeat(Shape desiredShape);

    @Override
    PtNDArray dot(NDArray other);

    @Override
    NDArray matMul(NDArray other);

    @Override
    PtNDArray clip(Number min, Number max);

    @Override
    PtNDArray swapAxes(int axis1, int axis2);

    @Override
    NDArray flip(int... axes);

    @Override
    PtNDArray transpose();

    @Override
    PtNDArray transpose(int... axes);

    @Override
    PtNDArray broadcast(Shape shape);

    @Override
    PtNDArray argMax();

    @Override
    PtNDArray argMax(int axis);

    @Override
    PtNDArray argMin();

    @Override
    PtNDArray argMin(int axis);

    @Override
    PtNDArray percentile(Number percentile);

    @Override
    PtNDArray percentile(Number percentile, int[] axes);

    @Override
    PtNDArray median();

    @Override
    PtNDArray median(int[] axes);

    @Override
    PtNDArray toDense();

    @Override
    PtNDArray toSparse(SparseFormat fmt);

    @Override
    PtNDArray nonzero();

    @Override
    PtNDArray erfinv();

    @Override
    PtNDArray inverse();

    @Override
    NDArray norm(boolean keepDims);

    @Override
    NDArray norm(int order, int[] axes, boolean keepDims);

    @Override
    NDArray oneHot(int depth);

    @Override
    NDArray oneHot(int depth, DataType dataType);

    @Override
    NDArray oneHot(int depth, float onValue, float offValue, DataType dataType);

    @Override
    NDArray batchDot(NDArray other);

    @Override
    PtNDArrayEx getNDArrayInternal();

    @Override
    String toString();

    @Override
    boolean equals(Object obj);

    @Override
    int hashCode();

    @Override
    void close();
}
