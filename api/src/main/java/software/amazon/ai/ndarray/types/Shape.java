/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package software.amazon.ai.ndarray.types;

import java.util.Arrays;

/** A class presents {@link software.amazon.ai.ndarray.NDArray}'s shape information. */
public class Shape {

    private int[] shape;

    /**
     * Constructs and initialize a <code>Shape</code> with specified dimension as {@code (int...
     * shape)}.
     *
     * @param shape dimensions of the shape
     */
    public Shape(int... shape) {
        this.shape = shape;
    }

    /**
     * Returns dimensions of the <code>Shape</code>.
     *
     * @return dimensions of the <code>Shape</code>
     */
    public int[] getShape() {
        return shape;
    }

    public long[] getShapeLong() {
        return Arrays.stream(getShape()).mapToLong((i) -> (long) i).toArray();
    }

    /**
     * Returns size of a specific dimension or several specific dimensions.
     *
     * @param dimensions The dimension or dimensions to find the size of
     * @return size of specific dimension(s)
     * @throws IllegalArgumentException thrown if passed an invalid dimension
     */
    public int size(int... dimensions) {
        int total = 1;
        for (int d : dimensions) {
            if (d < 0 || d >= shape.length) {
                throw new IllegalArgumentException("Invalid dimension " + d);
            }
            total *= shape[d];
        }
        return total;
    }

    /**
     * Returns the total size .
     *
     * @return total size
     */
    public int size() {
        int total = 1;
        for (int v : shape) {
            total *= v;
        }
        return total;
    }

    /**
     * Returns number of dimensions of this <code>Shape</code>.
     *
     * @return number of dimensions of this <code>Shape</code>.
     */
    public int dimension() {
        return shape.length;
    }

    public Shape drop(int n) {
        return slice(n + 1, shape.length);
    }

    public Shape slice(int from, int end) {
        int size = end - from;
        int[] out = new int[size];
        System.arraycopy(shape, from, out, 0, size);
        return new Shape(out);
    }

    public int head() {
        return shape[0];
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append('(');
        for (int i = 0; i < shape.length; ++i) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(shape[i]);
        }
        sb.append(')');
        return sb.toString();
    }

    /**
     * Returns the number of trailing ones in the array shape.
     *
     * <p>For example, a rank 3 array with shape [10, 1, 1] would return 2 for this method
     *
     * @return Number of trailing ones in shape
     */
    public int getTrailingOnes() {
        for (int i = 0; i < shape.length; i++) {
            if (shape[shape.length - i - 1] != 1) {
                return i;
            }
        }
        return 0;
    }

    /**
     * Returns the number of leading ones in the array shape.
     *
     * <p>For example, a rank 3 array with shape [1, 10, 1] would return value 1 for this method
     *
     * @return Number of leading ones in shape
     */
    public int getLeadingOnes() {
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] != 1) {
                return i;
            }
        }
        return 0;
    }

    /**
     * Returns the number of columns in the matrix.
     *
     * @return the number of columns in the matrix
     * @throws IllegalStateException Thrown if not a 2D matrix
     */
    public int columns() {
        if (!isMatrix()) {
            throw new IllegalStateException("Not a Matrix");
        }
        return shape[1];
    }

    /**
     * Returns the number of rows in the matrix.
     *
     * @return the number of rows in the matrix
     * @throws IllegalStateException Thrown if not a 2D matrix
     */
    public int rows() {
        if (!isMatrix()) {
            throw new IllegalStateException("Not a Matrix");
        }
        return shape[0];
    }

    /**
     * Returns true matrix and the number of columns is 1.
     *
     * @return true if matrix and the number of columns is 1
     */
    public boolean isColumnVector() {
        return isMatrix() && columns() == 1 && size() > 1;
    }

    /**
     * Returns true if matrix and the number of rows is 1.
     *
     * @return true if matrix and the number of rows is 1
     */
    public boolean isRowVector() {
        return isMatrix() && rows() == 1 && size() > 1;
    }

    /**
     * Returns true if this NDArray is a vector matrix.
     *
     * @return whether this NDArray is a vector matrix
     */
    public boolean isVectorMatrix() {
        return isColumnVector() || isRowVector();
    }

    /**
     * Returns whether the matrix has the same rows and columns.
     *
     * @return true if the matrix has the same rows and columns false otherwise
     */
    public boolean isSquare() {
        return isMatrix() && columns() == rows();
    }

    /**
     * Returns true if the NDArray is a matrix.
     *
     * @return whether the NDArray is a matrix
     */
    public boolean isMatrix() {
        return dimension() == 2;
    }

    /**
     * Returns true if the NDArray is a scalar.
     *
     * @return whether the NDArray is a scalar
     */
    public boolean isScalar() {
        return dimension() == 0;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Shape shape1 = (Shape) o;
        return Arrays.equals(shape, shape1.shape);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(shape);
    }
}
