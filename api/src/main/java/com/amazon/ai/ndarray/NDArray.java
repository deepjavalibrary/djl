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
import com.amazon.ai.ndarray.types.Layout;
import com.amazon.ai.ndarray.types.Shape;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.Buffer;
import java.util.List;
import java.util.concurrent.locks.Condition;

public interface NDArray extends AutoCloseable {

    /**
     * Returns the encoding format of the NDArray, or null.
     *
     * @return the encoded NDArray.
     */
    byte[] getEncoded();

    void encode(OutputStream os) throws IOException;

    DataType getDataType();

    Context getContext();

    Shape getShape();

    Layout getLayout();

    DataDesc getDataDescriptor();

    void set(Buffer data);

    void set(List<Float> data);

    NDArray at(int index);

    NDArray slice(int begin, int end);

    void copyTo(NDArray array);

    NDArray asInContext(Context ctx, boolean copy);

    void waitToRead();

    void waitToWrite();

    void waitAll();

    NDArray argsort(int axis, boolean isAscend);

    NDArray softmax(Integer axis, Double temperature);

    NDArray[] split(int numOutputs, Integer axis, Boolean squeezeAxis);

    boolean isSparse();

    boolean isCompressed();

    void markAsCompressed(boolean reallyCompressed);

    /**
     * Calculate the stride along a particular dimension
     *
     * @param dimension the dimension to get the stride for
     * @return the stride for a particular dimension
     */
    int stride(int dimension);

    /**
     * Element wise stride
     *
     * @return the element-wise stride for the array
     */
    int elementWiseStride();

    /**
     * Returns the number of possible vectors for a given dimension
     *
     * @param dimension the dimension to calculate the number of vectors for
     * @return the number of possible vectors along a dimension
     */
    long vectorsAlongDimension(int dimension);

    /**
     * Get the vector along a particular dimension
     *
     * @param index the index of the vector to getScalar
     * @param dimension the dimension to getScalar the vector from
     * @return the vector along a particular dimension
     */
    NDArray vectorAlongDimension(int index, int dimension);

    /**
     * Returns the number of possible vectors for a given dimension
     *
     * @param dimension the dimension to calculate the number of vectors for
     * @return the number of possible vectors along a dimension
     */
    long tensorsAlongDimension(int... dimension);

    /**
     * Get the vector along a particular dimension
     *
     * @param index the index of the vector to getScalar
     * @param dimension the dimension to getScalar the vector from
     * @return the vector along a particular dimension
     */
    NDArray tensorAlongDimension(int index, int... dimension);

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
     * Scalar addition (cloning)
     *
     * @param n the number to add
     * @return a clone with this matrix + the given number
     */
    NDArray add(Number n);

    /**
     * In place scalar addition
     *
     * @param n Number to add
     * @return This array, after adding value
     */
    NDArray addi(Number n);

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
     * Addition of this NDArray
     *
     * @param n the number to add
     * @param result the result NDArray
     * @return the result NDArray
     */
    NDArray add(Number n, NDArray result);

    /**
     * In place addition
     *
     * @param n the number to add
     * @param result the result NDArray
     * @return the result NDArray
     */
    NDArray addi(Number n, NDArray result);

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
     * Get the elements from this NDArray based on the specified indices
     *
     * @param indices an ndaray of the indices to get the elements for
     * @return the elements to get the array for
     */
    NDArray get(NDArray indices);

    /**
     * Get the elements from this NDArray based on the specified indices
     *
     * @param indices an ndaray of the indices to get the elements for
     * @return the elements to get the array for
     */
    NDArray get(List<List<Integer>> indices);

    /**
     * Get an NDArray comprised of the specified columns only. Copy operation.
     *
     * @param columns Columns to extract out of the current array
     * @return Array with only the specified columns
     */
    NDArray getColumns(int... columns);

    /**
     * Get an NDArray comprised of the specified rows only. Copy operation
     *
     * @param rows Rose to extract from this array
     * @return Array with only the specified rows
     */
    NDArray getRows(int... rows);

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
     * Get the linear index of the data in to the array
     *
     * @param i the index to getScalar
     * @return the linear index in to the data
     */
    long linearIndex(long i);

    /** @param list a list of NDArray */
    void sliceVectors(List<NDArray> list);

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
     * Insert a row in to this array Will throw an exception if this NDArray is not a matrix
     *
     * @param row the row insert into
     * @param toPut the row to insert
     * @return this
     */
    NDArray putRow(long row, NDArray toPut);

    /**
     * Insert a column in to this array Will throw an exception if this NDArray is not a matrix
     *
     * @param column the column to insert
     * @param toPut the array to put
     * @return this
     */
    NDArray putColumn(int column, NDArray toPut);

    /**
     * Returns the element at the specified row/column This will throw an exception if the
     *
     * @param row the row of the element to return
     * @param column the row of the element to return
     * @return a scalar NDArray of the element at this index
     */
    NDArray getScalar(long row, long column);

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
     * @param i the row insert into
     * @param j the column to insert into
     * @param element a scalar NDArray
     * @return a scalar NDArray of the element at this index
     */
    NDArray put(int i, int j, Number element);

    /**
     * Inserts the element at the specified index
     *
     * @param i the index insert into
     * @param element a scalar NDArray
     * @return a scalar NDArray of the element at this index
     */
    NDArray put(int i, NDArray element);

    /**
     * In place division of a column vector
     *
     * @param columnVector the column vector used for division
     * @return the result of the division
     */
    NDArray diviColumnVector(NDArray columnVector);

    /**
     * Division of a column vector (copy)
     *
     * @param columnVector the column vector used for division
     * @return the result of the division
     */
    NDArray divColumnVector(NDArray columnVector);

    /**
     * In place division of a row vector
     *
     * @param rowVector the row vector used for division
     * @return the result of the division
     */
    NDArray diviRowVector(NDArray rowVector);

    /**
     * Division of a row vector (copy)
     *
     * @param rowVector the row vector used for division
     * @return the result of the division
     */
    NDArray divRowVector(NDArray rowVector);

    /**
     * In place reverse divison of a column vector
     *
     * @param columnVector the column vector used for division
     * @return the result of the division
     */
    NDArray rdiviColumnVector(NDArray columnVector);

    /**
     * Reverse division of a column vector (copy)
     *
     * @param columnVector the column vector used for division
     * @return the result of the division
     */
    NDArray rdivColumnVector(NDArray columnVector);

    /**
     * In place reverse division of a column vector
     *
     * @param rowVector the row vector used for division
     * @return the result of the division
     */
    NDArray rdiviRowVector(NDArray rowVector);

    /**
     * Reverse division of a column vector (copy)
     *
     * @param rowVector the row vector used for division
     * @return the result of the division
     */
    NDArray rdivRowVector(NDArray rowVector);

    /**
     * In place multiplication of a column vector
     *
     * @param columnVector the column vector used for multiplication
     * @return the result of the multiplication
     */
    NDArray muliColumnVector(NDArray columnVector);

    /**
     * Multiplication of a column vector (copy)
     *
     * @param columnVector the column vector used for multiplication
     * @return the result of the multiplication
     */
    NDArray mulColumnVector(NDArray columnVector);

    /**
     * In place multiplication of a row vector
     *
     * @param rowVector the row vector used for multiplication
     * @return the result of the multiplication
     */
    NDArray muliRowVector(NDArray rowVector);

    /**
     * Multiplication of a row vector (copy)
     *
     * @param rowVector the row vector used for multiplication
     * @return the result of the multiplication
     */
    NDArray mulRowVector(NDArray rowVector);

    /**
     * In place reverse subtraction of a column vector
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    NDArray rsubiColumnVector(NDArray columnVector);

    /**
     * Reverse subtraction of a column vector (copy)
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    NDArray rsubColumnVector(NDArray columnVector);

    /**
     * In place reverse subtraction of a row vector
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    NDArray rsubiRowVector(NDArray rowVector);

    /**
     * Reverse subtraction of a row vector (copy)
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    NDArray rsubRowVector(NDArray rowVector);

    /**
     * In place subtraction of a column vector
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    NDArray subiColumnVector(NDArray columnVector);

    /**
     * Subtraction of a column vector (copy)
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    NDArray subColumnVector(NDArray columnVector);

    /**
     * In place subtraction of a row vector
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    NDArray subiRowVector(NDArray rowVector);

    /**
     * Subtraction of a row vector (copy)
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    NDArray subRowVector(NDArray rowVector);

    /**
     * In place addition of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    NDArray addiColumnVector(NDArray columnVector);

    /**
     * In place assignment of a column vector
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    NDArray putiColumnVector(NDArray columnVector);

    /**
     * Addition of a column vector (copy)
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    NDArray addColumnVector(NDArray columnVector);

    /**
     * In place addition of a row vector
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    NDArray addiRowVector(NDArray rowVector);

    /**
     * in place assignment of row vector, to each row of this array
     *
     * @param rowVector Row vector to put
     * @return This array, after assigning every road to the specified value
     */
    NDArray putiRowVector(NDArray rowVector);

    /**
     * Addition of a row vector (copy)
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    NDArray addRowVector(NDArray rowVector);

    /**
     * Perform a copy matrix multiplication
     *
     * @param other the other matrix to perform matrix multiply with
     * @return the result of the matrix multiplication
     */
    NDArray mmul(NDArray other);

    /**
     * Convert this NDArray to a 2d double matrix. Note that THIS SHOULD NOT BE USED FOR SPEED. This
     * is mainly used for integrations with other libraries. Due to nd4j's off heap nature, moving
     * data on heap is very expensive and should not be used if possible.
     *
     * @return a copy of this array as a 2d double array
     */
    double[][] toDoubleMatrix();

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
     * Convert this NDArray to a 2d float matrix. Note that THIS SHOULD NOT BE USED FOR SPEED. This
     * is mainly used for integrations with other libraries. Due to nd4j's off heap nature, moving
     * data on heap is very expensive and should not be used if possible.
     *
     * @return a copy of this array as a 2d float array
     */
    float[][] toFloatMatrix();

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
     * Convert this NDArray to a 2d int matrix. Note that THIS SHOULD NOT BE USED FOR SPEED. This is
     * mainly used for integrations with other libraries. Due to nd4j's off heap nature, moving data
     * on heap is very expensive and should not be used if possible.
     *
     * @return a copy of this array as a 2d int array
     */
    long[][] toLongMatrix();

    /**
     * Convert this NDArray to a 2d int matrix. Note that THIS SHOULD NOT BE USED FOR SPEED. This is
     * mainly used for integrations with other libraries. Due to nd4j's off heap nature, moving data
     * on heap is very expensive and should not be used if possible.
     *
     * @return a copy of this array as a 2d int array
     */
    int[][] toIntMatrix();

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
     * Element-wise copy addition of two NDArrays
     *
     * @param other the second NDArray to add
     * @return the result of the addition
     */
    NDArray add(NDArray other);

    /**
     * Element-wise copy addition of two NDArrays
     *
     * @param other the second NDArray to add
     * @param result the result NDArray
     * @return the result of the addition
     */
    NDArray add(NDArray other, NDArray result);

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
     * in place (element wise) addition of two NDArrays
     *
     * @param other the second NDArray to add
     * @return the result of the addition
     */
    NDArray addi(NDArray other);

    /**
     * in place (element wise) addition of two NDArrays
     *
     * @param other the second NDArray to add
     * @param result the result NDArray
     * @return the result of the addition
     */
    NDArray addi(NDArray other, NDArray result);

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
     * Returns the product along a given dimension
     *
     * @param dimension the dimension to getScalar the product along
     * @return the product along the specified dimension
     */
    NDArray prod(int... dimension);

    /**
     * Calculate the product of all values in the array
     *
     * @return Product of all values in the array
     */
    Number prodNumber();

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
     * This method takes boolean condition, and returns number of elements matching this condition
     *
     * @param condition Condition to calculate matches for
     * @return Number of elements matching condition
     */
    Number scan(Condition condition);

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
     * Returns a sub-NDArray.
     *
     * @param offsets offset
     * @param shape shape
     * @param stride stride
     * @return NDArray
     */
    NDArray subArray(long[] offsets, int[] shape, int[] stride);

    /**
     * Returns the elements at the specified indices
     *
     * @param indices the indices to getScalar
     * @return the array with the specified elements
     */
    NDArray getScalar(int... indices);

    NDArray getScalar(long... indices);

    /**
     * Get an integer value at the specified indices. Result will be cast to an integer, precision
     * loss is possible.
     *
     * @param indices Indices to get the integer at. Number of indices must match the array rank.
     * @return Integer value at the specified index
     */
    long getLong(int... indices);

    long getLong(long... indices);

    /**
     * Get a double value at the specified indices.
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
     * Returns a flattened version (row vector) of this NDArray
     *
     * @return a flattened version (row vector) of this NDArray
     */
    NDArray ravel();

    /**
     * Returns a flattened version (row vector) of this NDArray
     *
     * @param order the order of the new array
     * @return a flattened version (row vector) of this NDArray
     */
    NDArray ravel(char order);

    /**
     * Returns the number of slices in this NDArray
     *
     * @return the number of slices in this NDArray
     */
    long slices();

    /**
     * Get the number of trailing ones in the array shape. For example, a rank 3 array with shape
     * [10, 1, 1] would return 2 for this method
     *
     * @return Number of trailing ones in shape
     */
    int getTrailingOnes();

    /**
     * Get the number of leading ones in the array shape. For example, a rank 3 array with shape [1,
     * 10, 1] would return value 1 for this method
     *
     * @return Number of leading ones in shape
     */
    int getLeadingOnes();

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
     * Returns the start of where the NDArray is for the underlying data
     *
     * @return the starting offset
     */
    long offset();

    /**
     * Returns the start of where the NDArray is for the original data buffer
     *
     * @return NDArray NDArray
     */
    long originalOffset();

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
     * @param order the order of the new array
     * @param rows the rows of the matrix
     * @param columns the columns of the matrix
     * @return the reshaped NDArray
     */
    NDArray reshape(char order, int rows, int columns);

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
     * Flip the rows and columns of a matrix
     *
     * @return the flipped rows and columns of a matrix
     */
    NDArray transpose();

    /**
     * Flip the rows and columns of a matrix, in-place
     *
     * @return the flipped rows and columns of a matrix
     */
    NDArray transposei();

    /**
     * Mainly here for people coming from numpy. This is equivalent to a call to permute
     *
     * @param dimension the dimension to swap
     * @param with the one to swap it with
     * @return the swapped axes view
     */
    NDArray swapAxes(int dimension, int with);

    /**
     * See: http://www.mathworks.com/help/matlab/ref/permute.html
     *
     * @param rearrange the dimensions to swap to
     * @return the newly permuted array
     */
    NDArray permute(int... rearrange);

    /**
     * An <b>in-place</b> version of permute. The array shape information (shape, strides) is
     * modified by this operation (but not the data itself) See:
     * http://www.mathworks.com/help/matlab/ref/permute.html
     *
     * @param rearrange the dimensions to swap to
     * @return the current array
     */
    NDArray permutei(int... rearrange);

    /**
     * Dimshuffle: an extension of permute that adds the ability to broadcast various dimensions.
     * This will only accept integers and xs.
     *
     * <p>An x indicates a dimension should be broadcasted rather than permuted.
     *
     * <p>Examples originally from the theano docs:
     * http://deeplearning.net/software/theano/library/tensor/basic.html
     *
     * <p>Returns a view of this tensor with permuted dimensions. Typically the pattern will include
     * the integers 0, 1, ... ndim-1, and any number of 'x' characters in dimensions where this
     * tensor should be broadcasted.
     *
     * <p>A few examples of patterns and their effect:
     *
     * <p>('x') -&gt; make a 0d (scalar) into a 1d vector (0, 1) -&gt; identity for 2d vectors (1,
     * 0) -&gt; inverts the first and second dimensions ('x', 0) -&gt; make a row out of a 1d vector
     * (N to 1xN) (0, 'x') -&gt; make a column out of a 1d vector (N to Nx1) (2, 0, 1) -&gt; AxBxC
     * to CxAxB (0, 'x', 1) -&gt; AxB to Ax1xB (1, 'x', 0) -&gt; AxB to Bx1xA (1,) -&gt; This remove
     * dimensions 0. It must be a broadcastable dimension (1xA to A)
     *
     * @param rearrange the dimensions to swap to
     * @param newOrder the new order (think permute)
     * @param broadCastable (whether the dimension is broadcastable) (must be same length as new
     *     order)
     * @return the newly permuted array
     */
    NDArray dimShuffle(Object[] rearrange, int[] newOrder, boolean[] broadCastable);

    NDArray dimShuffle(Object[] rearrange, long[] newOrder, boolean[] broadCastable);

    /**
     * Returns the specified column. Throws an exception if its not a matrix
     *
     * @param i the column to getScalar
     * @return the specified column
     */
    NDArray getColumn(long i);

    /**
     * Returns the specified row. Throws an exception if its not a matrix
     *
     * @param i the row to getScalar
     * @return the specified row
     */
    NDArray getRow(long i);

    /**
     * Returns the number of columns in this matrix (throws exception if not 2d)
     *
     * @return the number of columns in this matrix
     */
    int columns();

    /**
     * Returns the number of rows in this matrix (throws exception if not 2d)
     *
     * @return the number of rows in this matrix
     */
    int rows();

    /**
     * Returns true if the number of columns is 1
     *
     * @return true if the number of columns is 1
     */
    boolean isColumnVector();

    /**
     * Returns true if the number of rows is 1
     *
     * @return true if the number of rows is 1
     */
    boolean isRowVector();

    /**
     * Returns true if the number of columns is 1
     *
     * @return true if the number of columns is 1
     */
    boolean isColumnVectorOrScalar();

    /**
     * Returns true if the number of rows is 1
     *
     * @return true if the number of rows is 1
     */
    boolean isRowVectorOrScalar();

    /**
     * Returns true if this NDArray is a vector
     *
     * @return whether this NDArray is a vector
     */
    boolean isVector();

    /**
     * Returns true if this NDArray is a vector or scalar
     *
     * @return whether this NDArray is a vector or scalar
     */
    boolean isVectorOrScalar();

    /**
     * Returns whether the matrix has the same rows and columns
     *
     * @return true if the matrix has the same rows and columns false otherwise
     */
    boolean isSquare();

    /**
     * Returns true if this NDArray is a matrix
     *
     * @return whether this NDArray is a matrix
     */
    boolean isMatrix();

    /**
     * Returns true if this NDArray is a scalar
     *
     * @return whether this NDArray is a scalar
     */
    boolean isScalar();

    /**
     * Returns the stride of this NDArray
     *
     * @return the stride of this NDArray
     */
    long[] stride();

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
    long length();

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
     * This method returns True, if this NDArray instance is attached to some Workspace. False
     * otherwise.
     *
     * @return True if attached to workspace, false otherwise
     */
    boolean isAttached();

    /**
     * This method pulls this NDArray into current Workspace, or optionally detaches if no workspace
     * is present.<br>
     * That is:<br>
     * If current workspace is present/active, NDArray is migrated to it.<br>
     * If no current workspace is present/active, one of two things occur: 1. If detachOnNoWs arg is
     * true: if there is no current workspace, NDArray is detached 2. If detachOnNoWs arg is false:
     * this NDArray is returned as-is (no-op)
     *
     * @param detachOnNoWs If true: detach on no WS. If false and no workspace: return this.
     * @return Migrated NDArray
     */
    NDArray migrate(boolean detachOnNoWs);

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
    int nnz();

    int[] flags();

    int[] hiddenDimensions();

    int[] sparseOffsets();

    /**
     * This method returns true if this NDArray is special case: no-value NDArray
     *
     * @return true if this NDArray is empty
     */
    boolean isEmpty();

    /**
     * This method returns dtype for this NDArray
     *
     * @return DataType
     */
    DataType dataType();

    /**
     * This method checks if this NDArray instance is one of Real types
     *
     * @return true if data type is floating point, false otherwise
     */
    boolean isR();

    /**
     * This method checks if this NDArray instance is one of integer types
     *
     * @return true if this NDArray instance is one of integer types
     */
    boolean isZ();

    /**
     * This method checks if this NDArray instance has boolean type
     *
     * @return if this NDArray instance has boolean type
     */
    boolean isB();

    /**
     * This method checks if this NDArray instance has String type
     *
     * @return true if this NDArray instance has String type
     */
    boolean isS();

    /**
     * This method cast elements of this NDArray to new data type
     *
     * @param dataType <code>DataType</code> to be casted
     * @return NDArray
     */
    NDArray castTo(DataType dataType);

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
