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
package com.amazon.ai.ndarray.types;

/** A class presents {@link com.amazon.ai.ndarray.NDArray}'s shape information. */
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

    /**
     * Returns size of specific dimensions {@code x}.
     *
     * @param x dimension
     * @return size of specific dimensions {@code x}
     */
    public int get(int x) {
        return shape[x];
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
        return slice(n, shape.length);
    }

    public Shape slice(int from, int end) {
        int size = end - from;
        int[] out = new int[size];
        System.arraycopy(shape, from, out, 0, size);
        return new Shape(out);
    }

    public int product() {
        int total = 1;
        for (int v : shape) {
            total *= v;
        }
        return total;
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
}
