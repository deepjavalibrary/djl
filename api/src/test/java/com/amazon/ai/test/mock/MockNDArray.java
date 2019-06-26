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
package com.amazon.ai.test.mock;

import com.amazon.ai.Context;
import com.amazon.ai.ndarray.Matrix;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.internal.NDArrayEx;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Layout;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;
import com.amazon.ai.training.GradReq;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.Buffer;
import java.util.List;
import java.util.concurrent.locks.Condition;

public class MockNDArray implements NDArray {

    @Override
    public byte[] getEncoded() {
        return new byte[0];
    }

    @Override
    public void encode(OutputStream os) throws IOException {}

    @Override
    public NDFactory getFactory() {
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
    public Layout getLayout() {
        return null;
    }

    @Override
    public DataDesc getDataDescriptor() {
        return null;
    }

    @Override
    public void set(Buffer data) {}

    @Override
    public void set(List<Float> data) {}

    @Override
    public NDArray at(int index) {
        return null;
    }

    @Override
    public NDArray slice(int begin, int end) {
        return null;
    }

    @Override
    public void copyTo(NDArray array) {}

    @Override
    public NDArray asInContext(Context ctx, boolean copy) {
        return null;
    }

    @Override
    public NDArray asType(DataType dtype, boolean copy) {
        return null;
    }

    @Override
    public void attachGrad() {}

    @Override
    public void attachGrad(GradReq gradReq, SparseFormat sparseFormat) {}

    @Override
    public NDArray getGradient() {
        return null;
    }

    @Override
    public void backward() {}

    @Override
    public void backward(boolean retainGraph, boolean isTraining) {}

    @Override
    public void backward(NDArray outGrad, boolean retainGraph, boolean isTraining) {}

    @Override
    public NDArray argsort(int axis, boolean ascending) {
        return null;
    }

    @Override
    public NDArray softmax(int[] axes) {
        return null;
    }

    @Override
    public NDArray softmax(int[] axes, double temperature) {
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
    public NDArray zerosLike() {
        return null;
    }

    @Override
    public NDArray onesLike() {
        return null;
    }

    @Override
    public boolean isSparse() {
        return false;
    }

    @Override
    public NDArray cumsumi(int dimension) {
        return null;
    }

    @Override
    public NDArray cumsum(int dimension) {
        return null;
    }

    @Override
    public NDArray assign(NDArray arr) {
        return null;
    }

    @Override
    public NDArray assignIf(NDArray arr, Condition condition) {
        return null;
    }

    @Override
    public NDArray replaceWhere(NDArray arr, Condition condition) {
        return null;
    }

    @Override
    public NDArray putScalar(long value, long... dimension) {
        return null;
    }

    @Override
    public NDArray putScalar(double value, long... dimension) {
        return null;
    }

    @Override
    public NDArray putScalar(float value, long... dimension) {
        return null;
    }

    @Override
    public NDArray putScalar(int value, long... dimension) {
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
    public NDArray eq(Number other) {
        return null;
    }

    @Override
    public NDArray eq(NDArray other) {
        return null;
    }

    @Override
    public NDArray gt(Number other) {
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
    public NDArray gt(NDArray other) {
        return null;
    }

    @Override
    public NDArray gte(Number other) {
        return null;
    }

    @Override
    public NDArray lte(Number other) {
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
    public NDArray isInfinite() {
        return null;
    }

    @Override
    public NDArray isNaN() {
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
    public NDArray rdiv(Number n) {
        return null;
    }

    @Override
    public NDArray rdivi(Number n) {
        return null;
    }

    @Override
    public NDArray rsub(Number n) {
        return null;
    }

    @Override
    public NDArray rsubi(Number n) {
        return null;
    }

    @Override
    public NDArray div(Number n) {
        return null;
    }

    @Override
    public NDArray divi(Number n) {
        return null;
    }

    @Override
    public NDArray mul(Number n) {
        return null;
    }

    @Override
    public NDArray muli(Number n) {
        return null;
    }

    @Override
    public NDArray sub(Number n) {
        return null;
    }

    @Override
    public NDArray subi(Number n) {
        return null;
    }

    @Override
    public NDArray add(Number n) {
        return null;
    }

    @Override
    public NDArray addi(Number n) {
        return null;
    }

    @Override
    public NDArray addi(NDArray other) {
        return null;
    }

    @Override
    public NDArray rdiv(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray rdivi(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray rsub(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray rsubi(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray div(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray divi(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray mul(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray muli(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray sub(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray subi(Number n, NDArray result) {
        return null;
    }

    @Override
    public NDArray match(NDArray comp, Condition condition) {
        return null;
    }

    @Override
    public NDArray match(Number comp, Condition condition) {
        return null;
    }

    @Override
    public NDArray getWhere(NDArray comp, Condition condition) {
        return null;
    }

    @Override
    public NDArray getWhere(Number comp, Condition condition) {
        return null;
    }

    @Override
    public NDArray putWhere(NDArray comp, NDArray put, Condition condition) {
        return null;
    }

    @Override
    public NDArray putWhere(Number comp, NDArray put, Condition condition) {
        return null;
    }

    @Override
    public NDArray putWhereWithMask(NDArray mask, NDArray put) {
        return null;
    }

    @Override
    public NDArray putWhereWithMask(NDArray mask, Number put) {
        return null;
    }

    @Override
    public NDArray putWhere(Number comp, Number put, Condition condition) {
        return null;
    }

    @Override
    public NDArray get(NDArray indices) {
        return null;
    }

    @Override
    public NDArray get(List<List<Integer>> indices) {
        return null;
    }

    @Override
    public NDArray rdiv(NDArray other) {
        return null;
    }

    @Override
    public NDArray rdivi(NDArray other) {
        return null;
    }

    @Override
    public NDArray rdiv(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray rdivi(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray rsub(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray rsub(NDArray other) {
        return null;
    }

    @Override
    public NDArray rsubi(NDArray other) {
        return null;
    }

    @Override
    public NDArray rsubi(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray assign(Number value) {
        return null;
    }

    @Override
    public NDArray putSlice(int slice, NDArray put) {
        return null;
    }

    @Override
    public NDArray cond(Condition condition) {
        return null;
    }

    @Override
    public NDArray repmat(int... shape) {
        return null;
    }

    @Override
    public NDArray repeat(int dimension, long... repeats) {
        return null;
    }

    @Override
    public NDArray getScalar(long i) {
        return null;
    }

    @Override
    public double squaredDistance(NDArray other) {
        return 0;
    }

    @Override
    public double distance2(NDArray other) {
        return 0;
    }

    @Override
    public double distance1(NDArray other) {
        return 0;
    }

    @Override
    public NDArray put(List<List<Integer>> indices, NDArray element) {
        return null;
    }

    @Override
    public NDArray put(NDArray indices, NDArray element) {
        return null;
    }

    @Override
    public NDArray put(NDArray element, int... indices) {
        return null;
    }

    @Override
    public NDArray put(int i, NDArray element) {
        return null;
    }

    @Override
    public NDArray mmul(NDArray other) {
        return null;
    }

    @Override
    public double[] toDoubleArray() {
        return new double[0];
    }

    @Override
    public float[] toFloatArray() {
        return new float[0];
    }

    @Override
    public int[] toIntArray() {
        return new int[0];
    }

    @Override
    public long[] toLongArray() {
        return new long[0];
    }

    @Override
    public NDArray mmul(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray div(NDArray other) {
        return null;
    }

    @Override
    public NDArray div(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray mul(NDArray other) {
        return null;
    }

    @Override
    public NDArray mul(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray sub(NDArray other) {
        return null;
    }

    @Override
    public NDArray sub(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray add(NDArray other) {
        return null;
    }

    @Override
    public NDArray mmuli(NDArray other) {
        return null;
    }

    @Override
    public NDArray mmuli(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray divi(NDArray other) {
        return null;
    }

    @Override
    public NDArray divi(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray muli(NDArray other) {
        return null;
    }

    @Override
    public NDArray muli(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray subi(NDArray other) {
        return null;
    }

    @Override
    public NDArray subi(NDArray other, NDArray result) {
        return null;
    }

    @Override
    public NDArray normmax(int... dimension) {
        return null;
    }

    @Override
    public Number normmaxNumber() {
        return null;
    }

    @Override
    public NDArray norm2(int... dimension) {
        return null;
    }

    @Override
    public Number norm2Number() {
        return null;
    }

    @Override
    public NDArray norm1(int... dimension) {
        return null;
    }

    @Override
    public Number norm1Number() {
        return null;
    }

    @Override
    public NDArray std(int... dimension) {
        return null;
    }

    @Override
    public Number stdNumber() {
        return null;
    }

    @Override
    public NDArray std(boolean biasCorrected, int... dimension) {
        return null;
    }

    @Override
    public Number stdNumber(boolean biasCorrected) {
        return null;
    }

    @Override
    public NDArray mean(int... dimension) {
        return null;
    }

    @Override
    public NDArray mean(NDArray result, int... dimension) {
        return null;
    }

    @Override
    public NDArray amean(int... dimension) {
        return null;
    }

    @Override
    public Number meanNumber() {
        return null;
    }

    @Override
    public Number ameanNumber() {
        return null;
    }

    @Override
    public NDArray var(int... dimension) {
        return null;
    }

    @Override
    public NDArray var(boolean biasCorrected, int... dimension) {
        return null;
    }

    @Override
    public Number varNumber() {
        return null;
    }

    @Override
    public NDArray max(int... dimension) {
        return null;
    }

    @Override
    public NDArray amax(int... dimension) {
        return null;
    }

    @Override
    public Number maxNumber() {
        return null;
    }

    @Override
    public Number amaxNumber() {
        return null;
    }

    @Override
    public NDArray min(int... dimension) {
        return null;
    }

    @Override
    public NDArray amin(int... dimension) {
        return null;
    }

    @Override
    public Number minNumber() {
        return null;
    }

    @Override
    public Number aminNumber() {
        return null;
    }

    @Override
    public NDArray sum(int... dimension) {
        return null;
    }

    @Override
    public NDArray sum(boolean keepDims, int... dimension) {
        return null;
    }

    @Override
    public NDArray sum(NDArray result, int... dimension) {
        return null;
    }

    @Override
    public Number sumNumber() {
        return null;
    }

    @Override
    public Number entropyNumber() {
        return null;
    }

    @Override
    public Number shannonEntropyNumber() {
        return null;
    }

    @Override
    public Number logEntropyNumber() {
        return null;
    }

    @Override
    public NDArray entropy(int... dimension) {
        return null;
    }

    @Override
    public NDArray shannonEntropy(int... dimension) {
        return null;
    }

    @Override
    public NDArray logEntropy(int... dimension) {
        return null;
    }

    @Override
    public NDArray getScalar(int... indices) {
        return null;
    }

    @Override
    public NDArray getScalar(long... indices) {
        return null;
    }

    @Override
    public long getLong(int... indices) {
        return 0;
    }

    @Override
    public long getLong(long... indices) {
        return 0;
    }

    @Override
    public double getDouble(int... indices) {
        return 0;
    }

    @Override
    public double getDouble(long... indices) {
        return 0;
    }

    @Override
    public float getFloat(int... indices) {
        return 0;
    }

    @Override
    public float getFloat(long... indices) {
        return 0;
    }

    @Override
    public NDArray dup() {
        return null;
    }

    @Override
    public NDArray ravel() {
        return null;
    }

    @Override
    public NDArray ravel(char order) {
        return null;
    }

    @Override
    public NDArray slice(long i, int dimension) {
        return null;
    }

    @Override
    public NDArray slice(long i) {
        return null;
    }

    @Override
    public NDArray reshape(char order, long... newShape) {
        return null;
    }

    @Override
    public NDArray reshape(char order, int... newShape) {
        return null;
    }

    @Override
    public NDArray reshape(long... newShape) {
        return null;
    }

    @Override
    public NDArray reshape(int[] shape) {
        return null;
    }

    @Override
    public NDArray swapAxes(int dimension, int with) {
        return null;
    }

    @Override
    public NDArray transpose(int... dimensions) {
        return null;
    }

    @Override
    public NDArray transposei(int... dimensions) {
        return null;
    }

    @Override
    public long size(int dimension) {
        return 0;
    }

    @Override
    public long size() {
        return 0;
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
    public Object element() {
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

    @Override
    public NDArray remainder(NDArray denominator) {
        return null;
    }

    @Override
    public NDArray remainder(NDArray denominator, NDArray result) {
        return null;
    }

    @Override
    public NDArray remainder(Number denominator) {
        return null;
    }

    @Override
    public NDArray remainder(Number denominator, NDArray result) {
        return null;
    }

    @Override
    public NDArray remainderi(NDArray denominator) {
        return null;
    }

    @Override
    public NDArray remainderi(Number denominator) {
        return null;
    }

    @Override
    public NDArray fmod(NDArray denominator) {
        return null;
    }

    @Override
    public NDArray fmod(NDArray denominator, NDArray result) {
        return null;
    }

    @Override
    public NDArray fmod(Number denominator) {
        return null;
    }

    @Override
    public NDArray fmod(Number denominator, NDArray result) {
        return null;
    }

    @Override
    public NDArray fmodi(NDArray denominator) {
        return null;
    }

    @Override
    public NDArray fmodi(Number denominator) {
        return null;
    }

    @Override
    public NDArray argMax(int... dimension) {
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
    public NDArray castTo(DataType dataType) {
        return null;
    }

    @Override
    public Matrix asMatrix() {
        return null;
    }

    @Override
    public boolean all() {
        return false;
    }

    @Override
    public boolean any() {
        return false;
    }

    @Override
    public boolean none() {
        return false;
    }

    @Override
    public NDArray like() {
        return null;
    }

    @Override
    public NDArray ulike() {
        return null;
    }

    @Override
    public NDArrayEx getNDArrayInternal() {
        return null;
    }

    @Override
    public void close() {}
}
