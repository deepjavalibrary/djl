package software.amazon.ai.ndarray;

public interface Matrix extends NDArray {

    /**
     * Inserts a row in to this matrix.
     *
     * @param row the row insert into
     * @param toPut the row to insert
     * @return this
     */
    NDArray putRow(long row, NDArray toPut);

    /**
     * Insert a column in to this array.
     *
     * @param column the column to insert
     * @param toPut the array to put
     * @return this
     */
    NDArray putColumn(int column, NDArray toPut);

    /**
     * Returns the element at the specified row/column.
     *
     * @param row the row of the element to return
     * @param column the row of the element to return
     * @return a scalar NDArray of the element at this index
     */
    NDArray getScalar(long row, long column);

    /**
     * In place division of a column vector.
     *
     * @param columnVector the column vector used for division
     * @return the result of the division
     */
    NDArray diviColumnVector(NDArray columnVector);

    /**
     * Division of a column vector (copy).
     *
     * @param columnVector the column vector used for division
     * @return the result of the division
     */
    NDArray divColumnVector(NDArray columnVector);

    /**
     * In place division of a row vector.
     *
     * @param rowVector the row vector used for division
     * @return the result of the division
     */
    NDArray diviRowVector(NDArray rowVector);

    /**
     * Division of a row vector (copy).
     *
     * @param rowVector the row vector used for division
     * @return the result of the division
     */
    NDArray divRowVector(NDArray rowVector);

    /**
     * In place reverse divison of a column vector.
     *
     * @param columnVector the column vector used for division
     * @return the result of the division
     */
    NDArray rdiviColumnVector(NDArray columnVector);

    /**
     * Reverse division of a column vector (copy).
     *
     * @param columnVector the column vector used for division
     * @return the result of the division
     */
    NDArray rdivColumnVector(NDArray columnVector);

    /**
     * In place reverse division of a column vector.
     *
     * @param rowVector the row vector used for division
     * @return the result of the division
     */
    NDArray rdiviRowVector(NDArray rowVector);

    /**
     * Reverse division of a column vector (copy).
     *
     * @param rowVector the row vector used for division
     * @return the result of the division
     */
    NDArray rdivRowVector(NDArray rowVector);

    /**
     * In place multiplication of a column vector.
     *
     * @param columnVector the column vector used for multiplication
     * @return the result of the multiplication
     */
    NDArray muliColumnVector(NDArray columnVector);

    /**
     * Multiplication of a column vector (copy).
     *
     * @param columnVector the column vector used for multiplication
     * @return the result of the multiplication
     */
    NDArray mulColumnVector(NDArray columnVector);

    /**
     * In place multiplication of a row vector.
     *
     * @param rowVector the row vector used for multiplication
     * @return the result of the multiplication
     */
    NDArray muliRowVector(NDArray rowVector);

    /**
     * Multiplication of a row vector (copy).
     *
     * @param rowVector the row vector used for multiplication
     * @return the result of the multiplication
     */
    NDArray mulRowVector(NDArray rowVector);

    /**
     * In place reverse subtraction of a column vector.
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    NDArray rsubiColumnVector(NDArray columnVector);

    /**
     * Reverse subtraction of a column vector (copy).
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    NDArray rsubColumnVector(NDArray columnVector);

    /**
     * In place reverse subtraction of a row vector.
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    NDArray rsubiRowVector(NDArray rowVector);

    /**
     * Reverse subtraction of a row vector (copy).
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    NDArray rsubRowVector(NDArray rowVector);

    /**
     * In place subtraction of a column vector.
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    NDArray subiColumnVector(NDArray columnVector);

    /**
     * Subtraction of a column vector (copy).
     *
     * @param columnVector the column vector to subtract
     * @return the result of the subtraction
     */
    NDArray subColumnVector(NDArray columnVector);

    /**
     * In place subtraction of a row vector.
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    NDArray subiRowVector(NDArray rowVector);

    /**
     * Subtraction of a row vector (copy).
     *
     * @param rowVector the row vector to subtract
     * @return the result of the subtraction
     */
    NDArray subRowVector(NDArray rowVector);

    /**
     * In place addition of a column vector.
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    NDArray addiColumnVector(NDArray columnVector);

    /**
     * In place assignment of a column vector.
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    NDArray putiColumnVector(NDArray columnVector);

    /**
     * Addition of a column vector (copy).
     *
     * @param columnVector the column vector to add
     * @return the result of the addition
     */
    NDArray addColumnVector(NDArray columnVector);

    /**
     * In place addition of a row vector.
     *
     * @param rowVector the row vector to add
     * @return the result of the addition
     */
    NDArray addiRowVector(NDArray rowVector);

    /**
     * in place assignment of row vector, to each row of this array.
     *
     * @param rowVector row vector to put
     * @return This array, after assigning every road to the specified value
     */
    NDArray putiRowVector(NDArray rowVector);

    /**
     * Addition of a row vector (copy).
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
     * Get a new NDArray comprised of the specified columns only.
     *
     * @param columns Columns to extract out of the current array
     * @return Array with only the specified columns
     */
    NDArray getColumns(int... columns);

    /**
     * Get a new NDArray comprised of the specified rows only.
     *
     * @param rows rows to extract from this array
     * @return Array with only the specified rows
     */
    NDArray getRows(int... rows);

    /**
     * Inserts the element at the specified index.
     *
     * @param i the row insert into
     * @param j the column to insert into
     * @param element a scalar NDArray
     * @return a scalar NDArray of the element at this index
     */
    NDArray put(int i, int j, Number element);

    /**
     * Reshapes the NDArray (can't change the length of the NDArray).
     *
     * @param order the order of the new array
     * @param rows the rows of the matrix
     * @param columns the columns of the matrix
     * @return the reshaped NDArray
     */
    NDArray reshape(char order, int rows, int columns);

    /**
     * Flip the rows and columns of a matrix.
     *
     * @return the flipped rows and columns of a matrix
     */
    NDArray transpose();

    /**
     * Flip the rows and columns of a matrix, in-place.
     *
     * @return the flipped rows and columns of a matrix
     */
    NDArray transposei();

    /**
     * Convert this NDArray to a 2d double matrix.
     *
     * <p>Note that THIS SHOULD NOT BE USED FOR SPEED. This is mainly used for integrations with
     * other libraries. Due to the off heap nature, moving data on heap is very expensive and should
     * not be used if possible.
     *
     * @return a copy of this array as a 2d double array
     */
    double[][] toDoubleMatrix();

    /**
     * Convert this NDArray to a 2d float matrix.
     *
     * <p>Note that THIS SHOULD NOT BE USED FOR SPEED. This is mainly used for integrations with
     * other libraries. Due to the off heap nature, moving data on to the heap is very expensive and
     * should not be used if possible.
     *
     * @return a copy of this array as a 2d float array
     */
    float[][] toFloatMatrix();

    /**
     * Convert this NDArray to a 2d long matrix.
     *
     * <p>Note that THIS SHOULD NOT BE USED FOR SPEED. This is mainly used for integrations with
     * other libraries. Due to the off heap nature, moving data on to the heap is very expensive and
     * should not be used if possible.
     *
     * @return a copy of this array as a 2d long array
     */
    long[][] toLongMatrix();

    /**
     * Convert this NDArray to a 2d int matrix.
     *
     * <p>Note that THIS SHOULD NOT BE USED FOR SPEED. This is mainly used for integrations with
     * other libraries. Due to the off heap nature, moving data on to the heap is very expensive and
     * should not be used if possible.
     *
     * @return a copy of this array as a 2d int array
     */
    int[][] toIntMatrix();
}
