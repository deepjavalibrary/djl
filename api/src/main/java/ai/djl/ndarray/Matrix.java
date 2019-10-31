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
package ai.djl.ndarray;

/** An interface representing a 2-dimensional matrix. */
public interface Matrix extends NDArray {

    /**
     * Inserts a row into this matrix.
     *
     * @param row the row to insert
     * @param toPut the array to insert into
     * @return this matrix
     */
    NDArray putRow(long row, NDArray toPut);

    /**
     * Inserts a column into this array.
     *
     * @param column the column to insert
     * @param toPut the array to insert into
     * @return this array
     */
    NDArray putColumn(int column, NDArray toPut);

    /**
     * Returns the element at the specified row/column.
     *
     * @param row the row of the element to return
     * @param column the column of the element to return
     * @return a scalar NDArray of the element at this index
     */
    NDArray getScalar(long row, long column);

    /**
     * Divides a column vector in place.
     *
     * @param columnVector the column vector used for division
     * @return the result of the division
     */
    NDArray diviColumnVector(NDArray columnVector);

    /**
     * Divides a column vector (copy).
     *
     * @param columnVector the column vector used for division
     * @return the result of the division
     */
    NDArray divColumnVector(NDArray columnVector);

    /**
     * Divides a row vector in place.
     *
     * @param rowVector the row vector used for division
     * @return the result of the division
     */
    NDArray diviRowVector(NDArray rowVector);

    /**
     * Divides a row vector (copy).
     *
     * @param rowVector the row vector used for division
     * @return the result of the division
     */
    NDArray divRowVector(NDArray rowVector);

    /**
     * Reverses the divison of a column vector in place.
     *
     * @param columnVector the column vector used for division
     * @return the result of the division
     */
    NDArray rdiviColumnVector(NDArray columnVector);

    /**
     * Reverses the division of a column vector (copy).
     *
     * @param columnVector the column vector used for division
     * @return the result of the division
     */
    NDArray rdivColumnVector(NDArray columnVector);

    /**
     * Reverses division of a column vector in place.
     *
     * @param rowVector the row vector used for division
     * @return the result of the division
     */
    NDArray rdiviRowVector(NDArray rowVector);

    /**
     * Reverses division of a column vector (copy).
     *
     * @param rowVector the row vector used for division
     * @return the result of the division
     */
    NDArray rdivRowVector(NDArray rowVector);

    /**
     * Multiplies a column vector in place.
     *
     * @param columnVector the column vector used for multiplication
     * @return the result of the multiplication
     */
    NDArray muliColumnVector(NDArray columnVector);

    /**
     * Multiplies a column vector (copy).
     *
     * @param columnVector the column vector used for multiplication
     * @return the result of the multiplication
     */
    NDArray mulColumnVector(NDArray columnVector);

    /**
     * Multiplies a row vector in place.
     *
     * @param rowVector the row vector used for multiplication
     * @return the result of the multiplication
     */
    NDArray muliRowVector(NDArray rowVector);

    /**
     * Multiplies a row vector (copy).
     *
     * @param rowVector the row vector used for multiplication
     * @return the result of the multiplication
     */
    NDArray mulRowVector(NDArray rowVector);

    /**
     * Reverses subtraction of a column vector in place.
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    NDArray rsubiColumnVector(NDArray columnVector);

    /**
     * Reverses subtraction of a column vector (copy).
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    NDArray rsubColumnVector(NDArray columnVector);

    /**
     * Reverses subtraction of a row vector in place.
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    NDArray rsubiRowVector(NDArray rowVector);

    /**
     * Reverses subtraction of a row vector (copy).
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    NDArray rsubRowVector(NDArray rowVector);

    /**
     * Subtracts a column vector in place.
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    NDArray subiColumnVector(NDArray columnVector);

    /**
     * Subtracts a column vector (copy).
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    NDArray subColumnVector(NDArray columnVector);

    /**
     * Subtracts a row vector in place.
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    NDArray subiRowVector(NDArray rowVector);

    /**
     * Subtracts a row vector (copy).
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    NDArray subRowVector(NDArray rowVector);

    /**
     * Adds a column vector in place.
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    NDArray addiColumnVector(NDArray columnVector);

    /**
     * Assigns a column vector in place.
     *
     * @param columnVector the column vector to add
     * @return the result of the assignment
     */
    NDArray putiColumnVector(NDArray columnVector);

    /**
     * Adds a column vector (copy).
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    NDArray addColumnVector(NDArray columnVector);

    /**
     * Adds a row vector in place.
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    NDArray addiRowVector(NDArray rowVector);

    /**
     * Assigns a row vector to each row of this array in place.
     *
     * @param rowVector the row vector to assign
     * @return this array, after assigning every row to the specified value
     */
    NDArray putiRowVector(NDArray rowVector);

    /**
     * Adds a row vector (copy).
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    NDArray addRowVector(NDArray rowVector);

    /**
     * Returns the specified column.
     *
     * @param i the column to getScalar
     * @return the specified column
     */
    NDArray getColumn(long i);

    /**
     * Returns the specified row.
     *
     * @param i the row to getScalar
     * @return the specified row
     */
    NDArray getRow(long i);

    /**
     * Returns a new NDArray comprised of the specified columns only.
     *
     * @param columns the columns to extract out of the current array
     * @return an array with only the specified columns
     */
    NDArray getColumns(int... columns);

    /**
     * Returns a new NDArray comprised of the specified rows only.
     *
     * @param rows the rows to extract from this array
     * @return an array with only the specified rows
     */
    NDArray getRows(int... rows);

    /**
     * Inserts the element at the specified index.
     *
     * @param i the row to insert into
     * @param j the column to insert into
     * @param element a scalar NDArray
     * @return a scalar NDArray of the element at this index
     */
    NDArray put(int i, int j, Number element);

    /**
     * Reshapes the NDArray (it can't change the length of the NDArray).
     *
     * @param order the order of the new array
     * @param rows the rows of the matrix
     * @param columns the columns of the matrix
     * @return the reshaped NDArray
     */
    NDArray reshape(char order, int rows, int columns);

    /**
     * Converts this NDArray to a 2d double matrix.
     *
     * <p>Note that THIS SHOULD NOT BE USED FOR SPEED. This is mainly used for integrations with
     * other libraries. Due to the off heap nature, moving data on heap is very expensive and should
     * not be used if possible.
     *
     * @return a copy of this array as a 2d double array
     */
    double[][] toDoubleMatrix();

    /**
     * Converts this NDArray to a 2d float matrix.
     *
     * <p>Note that THIS SHOULD NOT BE USED FOR SPEED. This is mainly used for integrations with
     * other libraries. Due to the off heap nature, moving data on to the heap is very expensive and
     * should not be used if possible.
     *
     * @return a copy of this array as a 2d float array
     */
    float[][] toFloatMatrix();

    /**
     * Converts this NDArray to a 2d long matrix.
     *
     * <p>Note that THIS SHOULD NOT BE USED FOR SPEED. This is mainly used for integrations with
     * other libraries. Due to the off heap nature, moving data on to the heap is very expensive and
     * should not be used if possible.
     *
     * @return a copy of this array as a 2d long array
     */
    long[][] toLongMatrix();

    /**
     * Converts this NDArray to a 2d int matrix.
     *
     * <p>Note that THIS SHOULD NOT BE USED FOR SPEED. This is mainly used for integrations with
     * other libraries. Due to the off heap nature, moving data on to the heap is very expensive and
     * should not be used if possible.
     *
     * @return a copy of this array as a 2d int array
     */
    int[][] toIntMatrix();
}
