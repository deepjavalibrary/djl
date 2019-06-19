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
package com.amazon.ai.ndarray;

import com.amazon.ai.Context;
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

/**
 * An interface represents n-dimensional array.
 *
 * <p>NDArray is the core data structure for all mathematical computations. An NDArray represents a
 * multidimensional, fixed-size homogeneous array. It has very close behaviour with python package
 * Numpy with the addition of efficient computing.
 */
public interface NDArray extends AutoCloseable {

    /**
     * Returns the encoding format of the NDArray, or null.
     *
     * @return the encoded NDArray.
     */
    byte[] getEncoded();

    /**
     * Encode NDArray to an {@link OutputStream}.
     *
     * @param os OutputStream
     * @throws IOException for writing problems
     */
    void encode(OutputStream os) throws IOException;

    /**
     * Returns the {@link NDFactory} used to create the NDArray.
     *
     * @return {@link NDFactory}
     */
    NDFactory getFactory();

    /**
     * Returns the {@link DataType} of the NDArray.
     *
     * <p>{@link DataType} is a definition of precision level of the NDArray. All values inside the
     * same NDArray would have the same data type.
     *
     * @return {@link DataType}
     */
    DataType getDataType();

    /**
     * Returns the {@link Context} of the NDArray.
     *
     * <p>{@link Context} class contains the information where this NDArray stored in memory, like
     * CPU/GPU.
     *
     * @return {@link Context}
     */
    Context getContext();

    /**
     * Returns the {@link Shape} of the NDArray.
     *
     * <p>{@link Shape} defines how this NDArray represent in multi-dimension.
     *
     * @return Returns the {@link Shape} of the NDArray.
     */
    Shape getShape();

    /**
     * Returns the {@link Layout} of the NDArray.
     *
     * <p>{@link Layout} defines the meaning of each dimension in the array.
     *
     * @return {@link Layout}
     */
    Layout getLayout();

    /**
     * Returns the {@link DataDesc} of the NDArray.
     *
     * <p>{@link DataDesc} contains all information about NDArray, including {@link Context}, {@link
     * DataType}, {@link Shape}, {@link Layout}, {@link com.amazon.ai.ndarray.types.SparseFormat}.
     *
     * @return {@link Layout}
     */
    DataDesc getDataDescriptor();

    /**
     * Set the NDArray value from {@link Buffer}.
     *
     * @param data The input buffered data
     */
    void set(Buffer data);

    /**
     * Set the NDArray value from a list of float.
     *
     * @param data list of floats to set
     */
    void set(List<Float> data);

    /**
     * Get the certain layer from the first dimension of the NDArray.
     *
     * @param index the layer index of the first dimension
     * @return NDArray from the layer
     */
    NDArray at(int index);

    /**
     * Getting a segment of the current NDArray.
     *
     * <p>The segmentation is only applied to the first dimension of the NDArray
     *
     * @param begin The beginning point
     * @param end the Engine pointer
     * @return Segmented NDArray
     */
    NDArray slice(int begin, int end);

    /**
     * Copy the current NDArray value to the one passed in.
     *
     * @param array the NDArray prepared to be copied to
     */
    void copyTo(NDArray array);

    /**
     * Converting the NDArray to a different {@link Context}.
     *
     * @param ctx {@link Context} that prepared to be set
     * @param copy set True if you want to return a copy of the Existing NDArray.
     * @return NDArray with the {@link Context} being set to
     */
    NDArray asInContext(Context ctx, boolean copy);

    /**
     * Converting the NDArray to a different {@link DataType}.
     *
     * @param dtype {@link DataType} that prepared to be set
     * @param copy set True if you want to return a copy of the Existing NDArray
     * @return NDArray with the {@link DataType} being set to
     */
    NDArray asType(DataType dtype, boolean copy);

    /**
     * Attach a gradient buffer to this NDArray, so that `backward` can compute gradient with
     * respect to it.
     */
    void attachGrad();

    /**
     * Attach a gradient buffer to this NDArray, so that `backward` can compute gradient with
     * respect to it.
     *
     * @param gradReq {@link GradReq} How gradient will be accumulated.
     * @param sparseFormat {@link SparseFormat} The storage type of the gradient array. Defaults to
     *     the same stype of this NDArray.
     */
    void attachGrad(GradReq gradReq, SparseFormat sparseFormat);

    /**
     * Returns the gradient buffer attached to this NDArray.
     *
     * @return the gradient buffer attached to this NDArray.
     */
    NDArray getGradient();

    /** Compute the gradients of this NDArray w.r.t variables. */
    void backward();

    /**
     * Compute the gradients of this NDArray w.r.t variables.
     *
     * @param retainGraph Whether to retain the computation graph for another backward pass on the
     *     same graph. By default the computation history is cleared.
     * @param isTraining Whether to compute gradient for training or inference.
     */
    void backward(boolean retainGraph, boolean isTraining);

    /**
     * Compute the gradients of this NDArray w.r.t variables.
     *
     * @param outGrad Gradient with respect to head
     * @param retainGraph Whether to retain the computation graph for another backward pass on the
     *     same graph. By default the computation history is cleared.
     * @param isTraining Whether to compute gradient for training or inference.
     */
    void backward(NDArray outGrad, boolean retainGraph, boolean isTraining);

    NDArray argsort(int axis, boolean isAscend);

    NDArray softmax(Integer axis, Double temperature);

    NDList split(int numOutputs, Integer axis, Boolean squeezeAxis);

    /**
     * Return an array of zeros with the same {@link Shape}, {@link DataType} and {@link
     * SparseFormat} as the input array
     *
     * @return {@link NDArray} filled with zeros
     */
    NDArray zerosLike();

    /**
     * Return an array of ones with the same {@link Shape}, {@link DataType} and {@link
     * SparseFormat} as the input array
     *
     * @return {@link NDArray} filled with ones
     */
    NDArray onesLike();

    boolean isSparse();

    /**
     * Returns the cumulative sum along a dimension. In-place method.
     *
     * @param dimension the dimension to perform cumulative sum along.
     * @return this object.
     */
    NDArray cumsumi(int dimension);

    /**
     * Returns the cumulative sum along a dimension.
     *
     * @param dimension the dimension to perform cumulative sum along.
     * @return the cumulative sum along the specified dimension
     */
    NDArray cumsum(int dimension);

    /**
     * Assign all of the elements in the given ndarray to this ndarray
     *
     * @param arr the elements to assign
     * @return this
     */
    NDArray assign(NDArray arr);

    /**
     * Assign all elements from given ndarray that are matching given condition, ndarray to this
     * ndarray
     *
     * @param arr the elements to assign
     * @param condition Condition to apply
     * @return this
     */
    NDArray assignIf(NDArray arr, Condition condition);

    /**
     * Replaces all elements in this ndarray that are matching give condition, with corresponding
     * elements from given array
     *
     * @param arr Source array
     * @param condition Condition to apply
     * @return New array with values conditionally replaced
     */
    NDArray replaceWhere(NDArray arr, Condition condition);

    /**
     * Put the specified value at the specified indices in this array.
     *
     * @param value Value to put
     * @param dimension Dimensions
     * @return This NDArray
     */
    NDArray putScalar(long value, long... dimension);

    NDArray putScalar(double value, long... dimension);

    NDArray putScalar(float value, long... dimension);

    NDArray putScalar(int value, long... dimension);

    /**
     * Returns the binary ndarray for "Epsilon equals" comparison.
     *
     * @param other the number to compare.
     * @return the binary ndarray for "Epsilon equals" comparison.
     */
    NDArray eps(Number other);

    /**
     * Returns the binary ndarray for "Epsilon equals" comparison.
     *
     * @param other the ndarray to compare.
     * @return the binary ndarray for "Epsilon equals" comparison.
     */
    NDArray eps(NDArray other);

    /**
     * Returns the binary ndarray for "Equals" comparison.
     *
     * @param other the number to compare.
     * @return the binary ndarray for "Equals" comparison.
     */
    NDArray eq(Number other);

    /**
     * Returns the binary ndarray for "Equals" comparison.
     *
     * @param other the ndarray to compare.
     * @return the binary ndarray for "Equals" comparison.
     */
    NDArray eq(NDArray other);

    /**
     * Returns the binary ndarray for "Greater" comparison.
     *
     * @param other the number to compare.
     * @return the binary ndarray for "Greater" comparison.
     */
    NDArray gt(Number other);

    /**
     * Returns the binary ndarray for "Not equals" comparison.
     *
     * @param other the number to compare.
     * @return the binary ndarray for "Not equals" comparison.
     */
    NDArray neq(Number other);

    /**
     * Returns the binary ndarray for "Not equals" comparison.
     *
     * @param other the ndarray to compare.
     * @return the binary ndarray for "Not equals" comparison.
     */
    NDArray neq(NDArray other);

    /**
     * Returns the binary ndarray for "Greater Than" comparison.
     *
     * @param other the ndarray to compare.
     * @return the binary ndarray for "Greater Than" comparison.
     */
    NDArray gt(NDArray other);

    /**
     * Returns binary ndarray for "Greter or equals" comparison.
     *
     * @param other the number to compare.
     * @return binary ndarray for "Greter or equals" comparison.
     */
    NDArray gte(Number other);

    /**
     * Returns the binary ndarray for "Less or equals" comparison.
     *
     * @param other the number to compare.
     * @return the binary ndarray for "Less or equals" comparison.
     */
    NDArray lte(Number other);

    /**
     * Returns the binary ndarray for "Less" comparison.
     *
     * @param other the number to compare.
     * @return the binary ndarray for "Less" comparison.
     */
    NDArray lt(Number other);

    /**
     * Returns the binary ndarray for "Less" comparison.
     *
     * @param other the ndarray to compare.
     * @return the binary ndarray for "Less" comparison.
     */
    NDArray lt(NDArray other);

    /**
     * Returns the binary NDArray with value true where this array's entries are infinite, or false
     * where they are not infinite
     *
     * @return the binary array with value true if the array are infinite.
     */
    NDArray isInfinite();

    /**
     * Returns the binary NDArray with value true where this array's entries are NaN, or false where
     * they are not infinite
     *
     * @return the binary array with value true if the array are NaN.
     */
    NDArray isNaN();

    /**
     * Returns the ndarray negative (cloned)
     *
     * @return Array copy with all values negated
     */
    NDArray neg();

    /**
     * In place setting of the negative version of this ndarray
     *
     * @return This array with all values negated
     */
    NDArray negi();

    /**
     * Reverse division with a scalar - i.e., (n / thisArrayValues)
     *
     * @param n Value to use for reverse division
     * @return Copy of array after applying reverse division
     */
    NDArray rdiv(Number n);

    /**
     * In place reverse division - i.e., (n / thisArrayValues)
     *
     * @param n Value to use for reverse division
     * @return This array after applying reverse division
     */
    NDArray rdivi(Number n);

    /**
     * Reverse subtraction with duplicates - i.e., (n - thisArrayValues)
     *
     * @param n Value to use for reverse subtraction
     * @return Copy of array after reverse subtraction
     */
    NDArray rsub(Number n);

    /**
     * Reverse subtraction in place - i.e., (n - thisArrayValues)
     *
     * @param n Value to use for reverse subtraction
     * @return This array after reverse subtraction
     */
    NDArray rsubi(Number n);

    /**
     * Division by a number
     *
     * @param n Number to divide values by
     * @return Copy of array after division
     */
    NDArray div(Number n);

    /**
     * In place scalar division
     *
     * @param n Number to divide values by
     * @return This array, after applying division operation
     */
    NDArray divi(Number n);

    /**
     * Scalar multiplication (copy)
     *
     * @param n the number to multiply by
     * @return a copy of this ndarray multiplied by the given number
     */
    NDArray mul(Number n);

    /**
     * In place scalar multiplication
     *
     * @param n The number to multiply by
     * @return This array, after applying scaler multiplication
     */
    NDArray muli(Number n);

    /**
     * Scalar subtraction (copied)
     *
     * @param n the number to subtract by
     * @return Copy of this array after applying subtraction operation
     */
    NDArray sub(Number n);

    /**
     * In place scalar subtraction
     *
     * @param n Number to subtract
     * @return This array, after applying subtraction operation
     */
    NDArray subi(Number n);

    /**
     * Adds a number to each element of the array.
     *
     * @param n the number to add
     * @return Returns the result of the addition
     */
    default NDArray add(Number n) {
        return NDArrays.add(this, n);
    }

    /**
     * Adds a number to each element of the array.
     *
     * @param n the number to add
     * @param params optional params to the function
     * @return Returns the result of the addition
     */
    default NDArray add(Number n, NDFuncParams params) {
        return NDArrays.add(this, n, params);
    }

    /**
     * Adds (broadcasting) another NDArray to this NDArray.
     *
     * @param other the other NDArray to add
     * @return Returns the result of the addition
     */
    default NDArray add(NDArray other) {
        return NDArrays.add(this, other);
    }

    /**
     * Adds (broadcasting) another NDArray to this NDArray.
     *
     * @param other the other NDArray to add
     * @param params optional params to the function
     * @return Returns the result of the addition
     */
    default NDArray add(NDArray other, NDFuncParams params) {
        return NDArrays.add(this, other, params);
    }

    /**
     * Reverse division (number / ndarray)
     *
     * @param n the number to divide by
     * @param result Array to place the result in. Must match shape of this array
     * @return Result array
     */
    NDArray rdiv(Number n, NDArray result);

    /**
     * Reverse in place division
     *
     * @param n the number to divide by
     * @param result the result ndarray
     * @return the result ndarray
     */
    NDArray rdivi(Number n, NDArray result);

    /**
     * Reverse subtraction
     *
     * @param n the number to subtract by
     * @param result the result ndarray
     * @return NDArray
     */
    NDArray rsub(Number n, NDArray result);

    /**
     * Reverse in place subtraction
     *
     * @param n the number to subtract by
     * @param result the result ndarray
     * @return the result ndarray
     */
    NDArray rsubi(Number n, NDArray result);

    /**
     * Division of this NDArray
     *
     * @param n the number to divide by
     * @param result the result NDArray
     * @return NDArray NDArray
     */
    NDArray div(Number n, NDArray result);

    /**
     * In place division of this NDArray
     *
     * @param n the number to divide by
     * @param result the result NDArray
     * @return NDArray NDArray
     */
    NDArray divi(Number n, NDArray result);

    /**
     * Multiplication of this NDArray.
     *
     * @param n the number to divide by
     * @param result the result NDArray
     * @return NDArray NDArray
     */
    NDArray mul(Number n, NDArray result);

    /**
     * In place multiplication of this NDArray
     *
     * @param n the number to divide by
     * @param result the result NDArray
     * @return NDArray NDArray
     */
    NDArray muli(Number n, NDArray result);

    NDArray sub(Number n, NDArray result);

    /**
     * In place subtraction of this NDArray
     *
     * @param n the number to subtract by
     * @param result the result NDArray
     * @return the result NDArray
     */
    NDArray subi(Number n, NDArray result);

    /**
     * Return a mask on whether each element matches the given condition.
     *
     * @param comp the number to compare
     * @param condition condition
     * @return NDArray NDArray
     */
    NDArray match(NDArray comp, Condition condition);

    /**
     * Returns a mask on whether each element matches the given condition.
     *
     * @param comp the NDArray to compare
     * @param condition condition
     * @return NDArray NDArray
     */
    NDArray match(Number comp, Condition condition);

    /**
     * Return the element if it fulfills the condition in result array.
     *
     * @param comp the comparison array
     * @param condition the condition to apply
     * @return the array fulfilling the criteria
     */
    NDArray getWhere(NDArray comp, Condition condition);

    /**
     * Boolean indexing: Return the element if it fulfills the condition in result array
     *
     * @param comp the comparison array
     * @param condition the condition to apply
     * @return the array fulfilling the criteria
     */
    NDArray getWhere(Number comp, Condition condition);

    /**
     * Assign the element according to the comparison array
     *
     * @param comp the comparison array
     * @param put the elements to put
     * @param condition the condition for masking on
     * @return NDArray NDArray
     */
    NDArray putWhere(NDArray comp, NDArray put, Condition condition);

    /**
     * Assign the element according to the comparison array
     *
     * @param comp the comparison array
     * @param put the elements to put
     * @param condition the condition for masking on
     * @return NDArray NDArray
     */
    NDArray putWhere(Number comp, NDArray put, Condition condition);

    /**
     * Use a pre computed mask for assigning arrays
     *
     * @param mask the mask to use
     * @param put the array to put
     * @return the resulting array
     */
    NDArray putWhereWithMask(NDArray mask, NDArray put);

    /**
     * Use a pre computed mask for assigning arrays
     *
     * @param mask the mask to use
     * @param put the array to put
     * @return the resulting array
     */
    NDArray putWhereWithMask(NDArray mask, Number put);

    /**
     * Assign the element according to the comparison array
     *
     * @param comp the comparison array
     * @param put the elements to put
     * @param condition the condition for masking on
     * @return the resulting array
     */
    NDArray putWhere(Number comp, Number put, Condition condition);

    /**
     * Returns the elements from this NDArray based on the specified indices
     *
     * @param indices an ndaray of the indices to get the elements for
     * @return the elements to get the array for
     */
    NDArray get(NDArray indices);

    /**
     * Returns the elements from this NDArray based on the specified indices
     *
     * @param indices an ndaray of the indices to get the elements for
     * @return the elements to get the array for
     */
    NDArray get(List<List<Integer>> indices);

    /**
     * Reverse division, elements wise. i.e., other / this
     *
     * @param other the matrix to divide from
     * @return Copy of this array after performing element wise reverse division
     */
    NDArray rdiv(NDArray other);

    /**
     * Reverse divsion (in place). i.e., other / this
     *
     * @param other The matrix to divide from
     * @return This array after performing element wise reverse division
     */
    NDArray rdivi(NDArray other);

    /**
     * Reverse division
     *
     * @param other the matrix to subtract from
     * @param result the result NDArray
     * @return NDArray NDArray
     */
    NDArray rdiv(NDArray other, NDArray result);

    /**
     * Reverse division (in-place)
     *
     * @param other the other NDArray to subtract
     * @param result the result NDArray
     * @return the NDArray with the operation applied
     */
    NDArray rdivi(NDArray other, NDArray result);

    /**
     * Reverse subtraction
     *
     * @param other the matrix to subtract from
     * @param result the result NDArray
     * @return the resulting array
     */
    NDArray rsub(NDArray other, NDArray result);

    /**
     * Element-wise reverse subtraction (copy op). i.e., other - this
     *
     * @param other Other array to use in reverse subtraction
     * @return Copy of this array, after applying reverse subtraction
     */
    NDArray rsub(NDArray other);

    /**
     * Element-wise reverse subtraction (in the place op) - i.e., other - this
     *
     * @param other Other way to use in reverse subtraction operation
     * @return This array, after applying reverse subtraction
     */
    NDArray rsubi(NDArray other);

    /**
     * Reverse subtraction (in-place)
     *
     * @param other the other NDArray to subtract
     * @param result the result NDArray
     * @return the NDArray with the operation applied
     */
    NDArray rsubi(NDArray other, NDArray result);

    /**
     * Set all entries of the NDArray to the specified value
     *
     * @param value the value to assign
     * @return the NDArray with the values
     */
    NDArray assign(Number value);

    /**
     * Assigns the given matrix (put) to the specified slice
     *
     * @param slice the slice to assign
     * @param put the slice to applyTransformToDestination
     * @return this for chainability
     */
    NDArray putSlice(int slice, NDArray put);

    /**
     * Returns a binary NDArray with value 'true' if the element matches the specified condition and
     * 'false' otherwise
     *
     * @param condition Condition to apply
     * @return Copy of this array with values 0 (condition does not apply), or one (condition
     *     applies)
     */
    NDArray cond(Condition condition);

    /**
     * Replicate and tile array to fill out to the given shape
     *
     * @param shape the new shape of this NDArray
     * @return the shape to fill out to
     */
    NDArray repmat(int... shape);

    /**
     * Repeat elements along a specified dimension.
     *
     * @param dimension the dimension to repeat
     * @param repeats the number of elements to repeat on each element
     * @return NDArray NDArray
     */
    NDArray repeat(int dimension, long... repeats);

    /**
     * Returns the element at the specified index
     *
     * @param i the index of the element to return
     * @return a scalar NDArray of the element at this index
     */
    NDArray getScalar(long i);

    /**
     * Returns the square of the Euclidean distance.
     *
     * @param other the other array to measure squared distance
     * @return squared distance
     */
    double squaredDistance(NDArray other);

    /**
     * Returns the (euclidean) distance.
     *
     * @param other the other array to measure distance
     * @return distance
     */
    double distance2(NDArray other);

    /**
     * Returns the (1-norm) distance.
     *
     * @param other the other array to measure distance
     * @return distance
     */
    double distance1(NDArray other);

    /**
     * Put element in to the indices denoted by the indices NDArray. This is equivalent to:
     * a[indices] = element
     *
     * <p>in numpy.
     *
     * @param indices the indices to put
     * @param element the element array to put
     * @return this array
     */
    NDArray put(List<List<Integer>> indices, NDArray element);

    /**
     * Put element in to the indices denoted by the indices NDArray. This is equivalent to:
     * a[indices] = element
     *
     * <p>in numpy.
     *
     * @param indices the indices to put
     * @param element the element array to put
     * @return this array
     */
    NDArray put(NDArray indices, NDArray element);

    /**
     * Inserts the element at the specified index
     *
     * @param element a scalar NDArray
     * @param indices the indices to insert into
     * @return a scalar NDArray of the element at this index
     */
    NDArray put(NDArray element, int... indices);

    /**
     * Inserts the element at the specified index
     *
     * @param i the index insert into
     * @param element a scalar NDArray
     * @return a scalar NDArray of the element at this index
     */
    NDArray put(int i, NDArray element);

    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    NDArray mmul(NDArray other);

    /**
     * Convert this NDArray to a 1d double matrix. Note that THIS SHOULD NOT BE USED FOR SPEED. This
     * is mainly used for integrations with other libraries. Due to nd4j's off heap nature, moving
     * data on heap is very expensive and should not be used if possible.
     *
     * @return a copy of this array as a 1d double array
     */
    double[] toDoubleArray();

    /**
     * Convert this NDArray to a 1d float vector. Note that THIS SHOULD NOT BE USED FOR SPEED. This
     * is mainly used for integrations with other libraries. Due to nd4j's off heap nature, moving
     * data on heap is very expensive and should not be used if possible.
     *
     * @return a copy of this array as a 1d float array
     */
    float[] toFloatArray();

    /**
     * Convert this NDArray to a 1d int matrix. Note that THIS SHOULD NOT BE USED FOR SPEED. This is
     * mainly used for integrations with other libraries. Due to nd4j's off heap nature, moving data
     * on heap is very expensive and should not be used if possible.
     *
     * @return a copy of this array as a 1d int array
     */
    int[] toIntArray();

    long[] toLongArray();

    /**
     * Perform an copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @param result the result NDArray
     * @return the result of the matrix multiplication
     */
    NDArray mmul(NDArray other, NDArray result);

    /**
     * Copy (element wise) division of two NDArrays
     *
     * @param other the second NDArray to divide
     * @return the result of the divide
     */
    NDArray div(NDArray other);

    /**
     * copy (element wise) division of two NDArrays
     *
     * @param other the second NDArray to divide
     * @param result the result NDArray
     * @return the result of the divide
     */
    NDArray div(NDArray other, NDArray result);

    /**
     * copy (element wise) multiplication of two NDArrays
     *
     * @param other the second NDArray to multiply
     * @return the result of the addition
     */
    NDArray mul(NDArray other);

    /**
     * copy (element wise) multiplication of two NDArrays
     *
     * @param other the second NDArray to multiply
     * @param result the result NDArray
     * @return the result of the multiplication
     */
    NDArray mul(NDArray other, NDArray result);

    /**
     * copy subtraction of two NDArrays
     *
     * @param other the second NDArray to subtract
     * @return the result of the addition
     */
    NDArray sub(NDArray other);

    /**
     * copy subtraction of two NDArrays
     *
     * @param other the second NDArray to subtract
     * @param result the result NDArray
     * @return the result of the subtraction
     */
    NDArray sub(NDArray other, NDArray result);

    /**
     * Perform an inplace matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    NDArray mmuli(NDArray other);

    /**
     * Perform an inplace matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @param result the result NDArray
     * @return the result of the matrix multiplication
     */
    NDArray mmuli(NDArray other, NDArray result);

    /**
     * in place (element wise) division of two NDArrays
     *
     * @param other the second NDArray to divide
     * @return the result of the divide
     */
    NDArray divi(NDArray other);

    /**
     * in place (element wise) division of two NDArrays
     *
     * @param other the second NDArray to divide
     * @param result the result NDArray
     * @return the result of the divide
     */
    NDArray divi(NDArray other, NDArray result);

    /**
     * in place (element wise) multiplication of two NDArrays
     *
     * @param other the second NDArray to multiply
     * @return the result of the multiplication
     */
    NDArray muli(NDArray other);

    /**
     * in place (element wise) multiplication of two NDArrays
     *
     * @param other the second NDArray to multiply
     * @param result the result NDArray
     * @return the result of the multiplication
     */
    NDArray muli(NDArray other, NDArray result);

    /**
     * in place (element wise) subtraction of two NDArrays
     *
     * @param other the second NDArray to subtract
     * @return the result of the subtraction
     */
    NDArray subi(NDArray other);

    /**
     * in place (element wise) subtraction of two NDArrays
     *
     * @param other the second NDArray to subtract
     * @param result the result NDArray
     * @return the result of the subtraction
     */
    NDArray subi(NDArray other, NDArray result);

    /**
     * Returns the max norm (aka infinity norm, equal to the maximum absolute value) along the
     * specified dimension(s)
     *
     * @param dimension the dimension to the max norm along
     * @return Max norm along the specified dimension
     */
    NDArray normmax(int... dimension);

    /**
     * Return the max norm (aka infinity norm, equal to the maximum absolute value) for the entire
     * array
     *
     * @return Max norm for the entire array
     */
    Number normmaxNumber();

    /**
     * Returns the norm2 (L2 norm, sqrt(sum(x_i^2), also known as Euclidean norm) along the
     * specified dimension(s)
     *
     * @param dimension the dimension to getScalar the norm2 along
     * @return the norm2 along the specified dimension
     */
    NDArray norm2(int... dimension);

    /**
     * Return the norm2 (L2 norm, sqrt(sum(x_i^2), also known as Euclidean norm) for the entire
     * array
     *
     * @return L2 norm for the array
     */
    Number norm2Number();

    /**
     * Returns the norm1 (L1 norm, i.e., sum of absolute values; also known as Taxicab or Manhattan
     * norm) along the specified dimension
     *
     * @param dimension the dimension to getScalar the norm1 along
     * @return the norm1 along the specified dimension
     */
    NDArray norm1(int... dimension);

    /**
     * Calculate and return norm1 (L1 norm, i.e., sum of absolute values; also known as Taxicab or
     * Manhattan norm) for the entire array
     *
     * @return Norm 1 for the array
     */
    Number norm1Number();

    /**
     * Standard deviation of an NDArray along one or more dimensions
     *
     * @param dimension the dimension to getScalar the std along
     * @return the standard deviation along a particular dimension
     */
    NDArray std(int... dimension);

    /**
     * Calculate the standard deviation for the entire array
     *
     * @return NDArray NDArray
     */
    Number stdNumber();

    /**
     * Standard deviation of an NDArray along a dimension
     *
     * @param biasCorrected If true: bias corrected standard deviation. False: not bias corrected
     * @param dimension the dimension to getScalar the std along
     * @return the standard deviation along a particular dimension
     */
    NDArray std(boolean biasCorrected, int... dimension);

    /**
     * Calculate the standard deviation for the entire array, specifying whether it is bias
     * corrected or not
     *
     * @param biasCorrected If true: bias corrected standard deviation. False: not bias corrected
     * @return Standard dev
     */
    Number stdNumber(boolean biasCorrected);

    /**
     * Returns the overall mean of this NDArray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this NDArray
     */
    NDArray mean(int... dimension);

    /**
     * Returns the overall mean of this NDArray
     *
     * @param result the result array
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this NDArray
     */
    NDArray mean(NDArray result, int... dimension);

    /**
     * Returns the absolute overall mean of this NDArray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this NDArray
     */
    NDArray amean(int... dimension);

    /**
     * Returns the overall mean of this NDArray
     *
     * @return the mean along the specified dimension of this NDArray
     */
    Number meanNumber();

    /**
     * Returns the absolute overall mean of this NDArray
     *
     * @return the mean along the specified dimension of this NDArray
     */
    Number ameanNumber();

    /**
     * Returns the overall variance of this NDArray
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this NDArray
     */
    NDArray var(int... dimension);

    /**
     * Returns the overall variance of this NDArray
     *
     * @param biasCorrected boolean on whether to apply corrected bias
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this NDArray
     */
    NDArray var(boolean biasCorrected, int... dimension);

    /**
     * Returns the overall variance of all values in this NDArray
     *
     * @return variance
     */
    Number varNumber();

    /**
     * Returns the overall max of this NDArray along given dimensions
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this NDArray
     */
    NDArray max(int... dimension);

    /**
     * Returns the absolute overall max of this NDArray along given dimensions
     *
     * @param dimension the dimension to getScalar the mean along
     * @return the mean along the specified dimension of this NDArray
     */
    NDArray amax(int... dimension);

    /**
     * Returns maximum value in this NDArray
     *
     * @return maximum value
     */
    Number maxNumber();

    /**
     * Returns maximum (absolute) value in this NDArray
     *
     * @return Max absolute value
     */
    Number amaxNumber();

    /**
     * Returns the overall min of this NDArray
     *
     * @param dimension the dimension to getScalar the min along
     * @return the mean along the specified dimension of this NDArray
     */
    NDArray min(int... dimension);

    /**
     * Returns minimum (absolute) value in this NDArray, along the specified dimensions
     *
     * @param dimension the dimension to getScalar the absolute min along
     * @return Minimum absolute value
     */
    NDArray amin(int... dimension);

    /**
     * Returns min value in this NDArray
     *
     * @return Minimum value in the array
     */
    Number minNumber();

    /**
     * Returns absolute min value in this NDArray
     *
     * @return Absolute min value
     */
    Number aminNumber();

    /**
     * Returns the sum along the last dimension of this NDArray
     *
     * @param dimension the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this NDArray
     */
    NDArray sum(int... dimension);

    NDArray sum(boolean keepDims, int... dimension);

    /**
     * Returns the sum along the last dimension of this NDArray
     *
     * @param result result of this operation will be stored here
     * @param dimension the dimension to getScalar the sum along
     * @return the sum along the specified dimension of this NDArray
     */
    NDArray sum(NDArray result, int... dimension);

    /**
     * Sum the entire array
     *
     * @return Sum of array
     */
    Number sumNumber();

    /**
     * Returns entropy value for this NDArray
     *
     * @return NDArray NDArray
     */
    Number entropyNumber();

    /**
     * Returns non-normalized Shannon entropy value for this NDArray
     *
     * @return NDArray NDArray
     */
    Number shannonEntropyNumber();

    /**
     * Returns log entropy value for this NDArray
     *
     * @return NDArray NDArray
     */
    Number logEntropyNumber();

    /**
     * Returns entropy value for this NDArray along specified dimension(s)
     *
     * @param dimension the dimension to return the entropy for
     * @return NDArray NDArray
     */
    NDArray entropy(int... dimension);

    /**
     * Returns entropy value for this NDArray along specified dimension(s)
     *
     * @param dimension the dimension to return the shannonEntropy for
     * @return NDArray
     */
    NDArray shannonEntropy(int... dimension);

    /**
     * Returns entropy value for this NDArray along specified dimension(s)
     *
     * @param dimension the dimension to return the logEntropy for
     * @return NDArray
     */
    NDArray logEntropy(int... dimension);

    /**
     * Returns the elements at the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    NDArray getScalar(int... indices);

    NDArray getScalar(long... indices);

    /**
     * Returns an integer value at the specified indices. Result will be cast to an integer,
     * precision loss is possible.
     *
     * @param indices Indices to get the integer at. Number of indices must match the array rank.
     * @return Integer value at the specified index
     */
    long getLong(int... indices);

    long getLong(long... indices);

    /**
     * Returns a double value at the specified indices.
     *
     * @param indices Indices to get the double at. Number of indices must match the array rank.
     * @return Double value at the specified index
     */
    double getDouble(int... indices);

    double getDouble(long... indices);

    /**
     * Returns the elements at the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    float getFloat(int... indices);

    float getFloat(long... indices);

    /**
     * Returns a copy of this NDArray
     *
     * @return a copy of this NDArray
     */
    NDArray dup();

    /**
     * Returns a flattened version of this NDArray
     *
     * @return a flattened version of this NDArray
     */
    NDArray ravel();

    /**
     * Returns a flattened version of this NDArray
     *
     * @param order the order of the new array
     * @return a flattened version of this NDArray
     */
    NDArray ravel(char order);

    /**
     * Returns the specified slice of this NDArray
     *
     * @param i the index of the slice to return
     * @param dimension the dimension to return the slice for
     * @return the specified slice of this NDArray
     */
    NDArray slice(long i, int dimension);

    /**
     * Returns the specified slice of this NDArray
     *
     * @param i the index of the slice to return
     * @return the specified slice of this NDArray
     */
    NDArray slice(long i);

    /**
     * Reshapes the NDArray (can't change the length of the NDArray). Typically this will be a view,
     * unless reshaping without copying is impossible.
     *
     * @param order the order of the new array
     * @param newShape the new shape of the NDArray
     * @return the reshaped NDArray
     */
    NDArray reshape(char order, long... newShape);

    NDArray reshape(char order, int... newShape);

    /**
     * Reshapes the NDArray (can't change the length of the NDArray). Typically this will be a view,
     * unless reshaping without copying is impossible.
     *
     * @param newShape the new shape of the NDArray
     * @return the reshaped NDArray
     */
    NDArray reshape(long... newShape);

    NDArray reshape(int[] shape);

    /**
     * Mainly here for people coming from numpy. This is equivalent to a call to permute
     *
     * @param dimension the dimension to swap
     * @param with the one to swap it with
     * @return the swapped axes view
     */
    NDArray swapAxes(int dimension, int with);

    /**
     * Reorders the dimensions in the {@link NDArray}.
     *
     * <p>Specify the new order for the axis given. Use -1 to broadcast across a dimension.
     *
     * @param dimensions the dimensions to swap to
     * @return the newly permuted array
     */
    NDArray transpose(int... dimensions);

    /**
     * Reorders the dimensions in the {@link NDArray} in place.
     *
     * <p>Specify the new order for the axis given. Use -1 to broadcast across a dimension.
     *
     * @param dimensions the dimensions to swap to
     * @return the current array
     */
    NDArray transposei(int... dimensions);

    /**
     * Returns the size along a specified dimension
     *
     * @param dimension the dimension to return the size for
     * @return the size of the array along the specified dimension
     */
    long size(int dimension);

    /**
     * Returns the total number of elements in the NDArray
     *
     * @return the number of elements in the NDArray
     */
    long size();

    /**
     * Broadcasts this NDArray to be the specified shape
     *
     * @param shape the new shape of this NDArray
     * @return the broadcasted NDArray
     */
    NDArray broadcast(long... shape);

    /**
     * Broadcasts this NDArray to be the specified shape
     *
     * @param result the result array
     * @return the broadcasted NDArray
     */
    NDArray broadcast(NDArray result);

    /**
     * Returns a scalar (individual element) of a scalar NDArray
     *
     * @return the individual item in this NDArray
     */
    Object element();

    /**
     * This method checks 2 NDArrays equality with given eps
     *
     * @param o object to compare with
     * @param eps Epsilon value to use for the quality operation
     * @return NDArray NDArray
     */
    boolean equalsWithEps(Object o, double eps);

    /**
     * This method checks 2 NDArrays for equal shapes.<br>
     * Shapes are considered equal if:<br>
     * (a) Both arrays have equal rank, and<br>
     * (b) size(0)...size(rank()-1) are equal for both arrays
     *
     * @param other Other
     * @return True if shap
     */
    boolean equalShapes(NDArray other);

    /**
     * Remainder operator
     *
     * @param denominator the denominator
     * @return NDArray NDArray
     */
    NDArray remainder(NDArray denominator);

    /**
     * Remainder operator
     *
     * @param denominator the denominator
     * @param result the result array to put this in
     * @return NDArray NDArray
     */
    NDArray remainder(NDArray denominator, NDArray result);

    /**
     * The scalar denominator
     *
     * @param denominator the denominator as a scalar
     * @return NDArray NDArray
     */
    NDArray remainder(Number denominator);

    /**
     * @param denominator the denominator as a scalar
     * @param result the result array
     * @return NDArray NDArray
     */
    NDArray remainder(Number denominator, NDArray result);

    /**
     * In place remainder
     *
     * @param denominator the denominator as a scalar
     * @return NDArray NDArray
     */
    NDArray remainderi(NDArray denominator);

    /**
     * In place remainder
     *
     * @param denominator the denominator as a scalar
     * @return NDArray NDArray
     */
    NDArray remainderi(Number denominator);

    /**
     * remainder of division
     *
     * @param denominator the array of denominators for each element in this array
     * @return NDArray NDArray
     */
    NDArray fmod(NDArray denominator);

    /**
     * remainder of division
     *
     * @param denominator the denominator as a scalar
     * @param result the result array
     * @return NDArray NDArray
     */
    NDArray fmod(NDArray denominator, NDArray result);

    /**
     * @param denominator the denominator as a scalar
     * @return NDArray NDArray
     */
    NDArray fmod(Number denominator);

    NDArray fmod(Number denominator, NDArray result);

    /**
     * In place fmod
     *
     * @param denominator the denominator as a scalar
     * @return NDArray NDArray
     */
    NDArray fmodi(NDArray denominator);

    /**
     * In place fmod
     *
     * @param denominator the denominator as a scalar
     * @return NDArray NDArray
     */
    NDArray fmodi(Number denominator);

    /**
     * This method returns index of highest value along specified dimension(s)
     *
     * @param dimension Dimension along which to perform the argMax operation
     * @return Array containing indices
     */
    NDArray argMax(int... dimension);

    /**
     * This method returns percentile value for this NDArray
     *
     * @param percentile target percentile in range of 0..100
     * @return NDArray NDArray
     */
    Number percentileNumber(Number percentile);

    /**
     * This method returns median value for this NDArray
     *
     * @return Median value for array
     */
    Number medianNumber();

    /**
     * This method returns median along given dimension(s)
     *
     * @param dimension Dimension along which to perform the median operation
     * @return Median along specified dimensions
     */
    NDArray median(int... dimension);

    /**
     * This method returns median along given dimension(s)
     *
     * @param percentile target percentile in range of 0..100
     * @param dimension Dimension to calculate percentile for
     * @return NDArray NDArray
     */
    NDArray percentile(Number percentile, int... dimension);

    // ------------ Sparse methods ------------

    /**
     * Return a dense representation of the sparse NDArray
     *
     * @return NDArray NDArray
     */
    NDArray toDense();

    /**
     * Return the number of non-null element
     *
     * @return nnz
     */
    int nonzero();

    /**
     * This method returns true if this NDArray is special case: no-value NDArray
     *
     * @return true if this NDArray is empty
     */
    boolean isEmpty();

    /**
     * This method cast elements of this NDArray to new data type
     *
     * @param dataType <code>DataType</code> to be casted
     * @return NDArray
     */
    NDArray castTo(DataType dataType);

    /**
     * This method converts the array into a 2D Matrix.
     *
     * @return This NDArray as Matrix
     * @throws IllegalStateException Thrown if the NDArray is not a 2D matrix
     */
    Matrix asMatrix();

    /**
     * This method checks if all elements within this array are non-zero (or true, in case of
     * boolean)
     *
     * @return true if all elements within this array are non-zero
     */
    boolean all();

    /**
     * This method checks if any of the elements within this array are non-zero (or true, in case of
     * boolean)
     *
     * @return true if any of the elements within this array are non-zero
     */
    boolean any();

    /**
     * This method checks if any of the elements within this array are non-zero (or true, in case of
     * boolean)
     *
     * @return NDArray
     */
    boolean none();

    /**
     * This method returns empty array with the same dtype/order/shape as this one
     *
     * @return NDArray
     */
    NDArray like();

    /**
     * This method returns uninitialized array with the same dtype/order/shape as this one
     *
     * @return NDArray
     */
    NDArray ulike();

    @Override
    void close();
}
