package org.apache.mxnet.engine;

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
import java.util.List;
import java.util.concurrent.locks.Condition;

public class MxMatrix implements Matrix {

    private MxNDArray array;

    public MxMatrix(MxNDArray array) {
        this.array = array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putRow(long row, NDArray toPut) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putColumn(int column, NDArray toPut) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getScalar(long row, long column) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray diviColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray diviRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiviColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiviRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muliColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mulColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muliRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mulRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubiColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubiRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subiColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subiRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addiColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putiColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addColumnVector(NDArray columnVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addiRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray putiRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addRowVector(NDArray rowVector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getColumn(long i) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getRow(long i) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getColumns(int... columns) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getRows(int... rows) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray put(int i, int j, Number element) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(char order, int rows, int columns) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transposei() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public double[][] toDoubleMatrix() {
        return new double[0][];
    }

    /** {@inheritDoc} */
    @Override
    public float[][] toFloatMatrix() {
        return new float[0][];
    }

    /** {@inheritDoc} */
    @Override
    public long[][] toLongMatrix() {
        return new long[0][];
    }

    /** {@inheritDoc} */
    @Override
    public int[][] toIntMatrix() {
        return new int[0][];
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
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Context getContext() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Layout getLayout() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc getDataDescriptor() {
        return array.getDataDescriptor();
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {}

    /** {@inheritDoc} */
    @Override
    public void set(List<Float> data) {}

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
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int axis, boolean squeezeAxis, NDFuncParams fparams) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int axis, int numOutputs, NDFuncParams fparams)
            throws IllegalArgumentException {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike(NDFuncParams fparams) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike(NDFuncParams fparams) {
        return null;
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
        return new double[0];
    }

    /** {@inheritDoc} */
    @Override
    public float[] toFloatArray() {
        return new float[0];
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
    public long size(int dimension) {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public long size() {
        return 0;
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
        return this;
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
    public void close() {}
}
