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
package org.apache.mxnet.engine;

public class Shape {

    private int[] shape;

    public Shape(int... shape) {
        this.shape = shape;
    }

    public int[] getShape() {
        return shape;
    }

    public int get(int x) {
        return shape[x];
    }

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
