/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.nn.convolutional;

import ai.djl.ndarray.types.Shape;

public class TestData {

    private final int index;

    private int filters;
    private int depth;
    private int height;
    private int width;

    private Shape kernel;
    private Shape padding;
    private Shape outputPadding;
    private Shape stride;
    private Shape dilation;

    public TestData(int index) {
        this.index = index;
    }

    public int getIndex() {
        return index;
    }

    public int getFilters() {
        return filters;
    }

    public void setFilters(int filters) {
        this.filters = filters;
    }

    public int getDepth() {
        return depth;
    }

    public void setDepth(int depth) {
        this.depth = depth;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public Shape getKernel() {
        return kernel;
    }

    public void setKernel(Shape kernel) {
        this.kernel = kernel;
    }

    public Shape getPadding() {
        return padding;
    }

    public void setPadding(Shape padding) {
        this.padding = padding;
    }

    public Shape getOutputPadding() {
        return outputPadding;
    }

    public void setOutputPadding(Shape outputPadding) {
        this.outputPadding = outputPadding;
    }

    public Shape getStride() {
        return stride;
    }

    public void setStride(Shape stride) {
        this.stride = stride;
    }

    public Shape getDilation() {
        return dilation;
    }

    public void setDilation(Shape dilation) {
        this.dilation = dilation;
    }
}
