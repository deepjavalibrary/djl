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
package ai.djl.nn.pooling;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.util.Preconditions;
import java.util.Objects;

/**
 * Utility class that provides {@code Block} and methods for different pooling functions.
 *
 * @see <a href="https://d2l.djl.ai/chapter_convolutional-neural-networks/pooling.html">The D2L
 *     chapter on pooling</a>
 */
public final class Pool {

    private Pool() {}

    /**
     * Performs 1-D Max Pooling on the input.
     *
     * @param input the NDArray on which max pooling is performed
     * @param kernelShape the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param padding the padding to be set in each dimension
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the NDArray after applying max pooling
     */
    public static NDArray maxPool1d(
            NDArray input, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        Objects.requireNonNull(kernelShape, "kernelShape cannot be null for maxPool1d");
        Preconditions.checkArgument(
                input.getShape().dimension() == 3,
                "Expect input dimension is 3 but got " + input.getShape().dimension());
        Preconditions.checkArgument(
                kernelShape.dimension() == 1 && stride.dimension() == 1 && padding.dimension() == 1,
                "kernelShape, Stride and Padding dimensions for maxPool1d layer should be 1");
        return input.getNDArrayInternal().maxPool(kernelShape, stride, padding, ceilMode);
    }

    /**
     * Performs 2-D Max Pooling on the input.
     *
     * @param input the NDArray on which max pooling is performed
     * @param kernelShape the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param padding the padding to be set in each dimension
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the NDArray after applying max pooling
     */
    public static NDArray maxPool2d(
            NDArray input, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        Objects.requireNonNull(kernelShape, "kernelShape cannot be null for maxPool2d");
        Preconditions.checkArgument(
                input.getShape().dimension() == 4,
                "Expect input dimension is 4 but got " + input.getShape().dimension());
        Preconditions.checkArgument(
                kernelShape.dimension() == 2 && stride.dimension() == 2 && padding.dimension() == 2,
                "kernelShape, Stride and Padding dimensions for maxPool2d should be 2");
        return input.getNDArrayInternal().maxPool(kernelShape, stride, padding, ceilMode);
    }

    /**
     * Performs 3-D Max Pooling on the input.
     *
     * @param input the NDArray on which max pooling is performed
     * @param kernelShape the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param padding the padding to be set in each dimension
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the NDArray after applying max pooling
     */
    public static NDArray maxPool3d(
            NDArray input, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        Objects.requireNonNull(kernelShape, "kernelShape cannot be null for maxPool3d");
        Preconditions.checkArgument(
                input.getShape().dimension() == 5,
                "Expect input dimension is 5 but got " + input.getShape().dimension());
        Preconditions.checkArgument(
                kernelShape.dimension() == 3 && stride.dimension() == 3 && padding.dimension() == 3,
                "kernelShape, Stride and Pad dimensions for maxPool3d should be 3");
        return input.getNDArrayInternal().maxPool(kernelShape, stride, padding, ceilMode);
    }

    /**
     * Performs 1-D Global Max Pooling on the input.
     *
     * @param input the NDArray on which max pooling is performed
     * @return the NDArray after applying global max pooling
     */
    public static NDArray globalMaxPool1d(NDArray input) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 3,
                "Expect input dimension is 3 but got " + input.getShape().dimension());
        return input.getNDArrayInternal().globalMaxPool();
    }

    /**
     * Performs 2-D Global Max Pooling on the input.
     *
     * @param input the NDArray on which max pooling is performed
     * @return the NDArray after applying global max pooling
     */
    public static NDArray globalMaxPool2d(NDArray input) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 4,
                "Expect input dimension is 4 but got " + input.getShape().dimension());
        return input.getNDArrayInternal().globalMaxPool();
    }

    /**
     * Performs 3-D Global Max Pooling on the input.
     *
     * @param input the NDArray on which max pooling is performed
     * @return the NDArray after applying global max pooling
     */
    public static NDArray globalMaxPool3d(NDArray input) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 5,
                "Expect input dimension is 5 but got " + input.getShape().dimension());
        return input.getNDArrayInternal().globalMaxPool();
    }

    /**
     * Performs 1-D Avg Pooling on the input.
     *
     * @param input the NDArray on which average pooling is performed
     * @param kernelShape the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param padding the padding to be set in each dimension
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @param countIncludePad whether to include padding for calculations
     * @return the NDArray after applying avg pooling
     */
    public static NDArray avgPool1d(
            NDArray input,
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        Objects.requireNonNull(kernelShape, "kernelShape cannot be null for avgPool1d");
        Preconditions.checkArgument(
                input.getShape().dimension() == 3,
                "Expect input dimension is 3 but got " + input.getShape().dimension());
        Preconditions.checkArgument(
                kernelShape.dimension() == 1 && stride.dimension() == 1 && padding.dimension() == 1,
                "kernelShape, Stride and Padding dimensions for avgPool1d should be 1");
        return input.getNDArrayInternal()
                .avgPool(kernelShape, stride, padding, ceilMode, countIncludePad);
    }

    /**
     * Performs 2-D Avg Pooling on the input.
     *
     * @param input the NDArray on which average pooling is performed
     * @param kernelShape the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param padding the padding to be set in each dimension
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @param countIncludePad whether to include padding for calculations
     * @return the NDArray after applying avg pooling
     */
    public static NDArray avgPool2d(
            NDArray input,
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        Objects.requireNonNull(kernelShape, "kernelShape cannot be null for avgPool2d");
        Preconditions.checkArgument(
                input.getShape().dimension() == 4,
                "Expect input dimension is 4 but got " + input.getShape().dimension());
        Preconditions.checkArgument(
                kernelShape.dimension() == 2 && stride.dimension() == 2 && padding.dimension() == 2,
                "kernelShape, Stride and Padding dimensions for avgPool2d should be 2");
        return input.getNDArrayInternal()
                .avgPool(kernelShape, stride, padding, ceilMode, countIncludePad);
    }

    /**
     * Performs 3-D Avg Pooling on the input.
     *
     * @param input the NDArray on which average pooling is performed
     * @param kernelShape the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param padding the padding to be set in each dimension
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @param countIncludePad whether to include padding for calculations
     * @return the NDArray after applying avg pooling
     */
    public static NDArray avgPool3d(
            NDArray input,
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        Objects.requireNonNull(kernelShape, "kernelShape cannot be null for avgPool3d");
        Preconditions.checkArgument(
                input.getShape().dimension() == 5,
                "Expect input dimension is 5 but got " + input.getShape().dimension());
        Preconditions.checkArgument(
                kernelShape.dimension() == 3 && stride.dimension() == 3 && padding.dimension() == 3,
                "kernelShape, Stride and Padding dimensions for avgPool2d should be 3");
        return input.getNDArrayInternal()
                .avgPool(kernelShape, stride, padding, ceilMode, countIncludePad);
    }

    /**
     * Performs 1-D Global Avg Pooling on the input.
     *
     * @param input the NDArray on which average pooling is performed
     * @return the NDArray after applying global avg pooling
     */
    public static NDArray globalAvgPool1d(NDArray input) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 3,
                "Expect input dimension is 3 but got " + input.getShape().dimension());
        return input.getNDArrayInternal().globalAvgPool();
    }

    /**
     * Performs 2-D Global Avg Pooling on the input.
     *
     * @param input the NDArray on which average pooling is performed
     * @return the NDArray after applying global avg pooling
     */
    public static NDArray globalAvgPool2d(NDArray input) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 4,
                "Expect input dimension is 4 but got " + input.getShape().dimension());
        return input.getNDArrayInternal().globalAvgPool();
    }

    /**
     * Performs 3-D Global Avg Pooling on the input.
     *
     * @param input the NDArray on which average pooling is performed
     * @return the NDArray after applying global avg pooling
     */
    public static NDArray globalAvgPool3d(NDArray input) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 5,
                "Expect input dimension is 5 but got " + input.getShape().dimension());
        return input.getNDArrayInternal().globalAvgPool();
    }

    /**
     * Performs 1-D LP Pooling on the input.
     *
     * @param input the NDArray on which LP pooling is performed
     * @param normType float value indicating norm
     * @param kernelShape the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param padding the padding to be set in each dimension
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the NDArray after applying lp pooling
     */
    public static NDArray lpPool1d(
            NDArray input,
            float normType,
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode) {
        Objects.requireNonNull(kernelShape, "kernelShape cannot be null for lpPool1d");
        Preconditions.checkArgument(
                input.getShape().dimension() == 3,
                "Expect input dimension is 3 but got " + input.getShape().dimension());
        Preconditions.checkArgument(
                kernelShape.dimension() == 1 && stride.dimension() == 1 && padding.dimension() == 1,
                "kernelShape, Stride and Padding dimensions for lpPool1d should be 1");
        return input.getNDArrayInternal().lpPool(normType, kernelShape, stride, padding, ceilMode);
    }

    /**
     * Performs 2-D LP Pooling on the input.
     *
     * @param input the NDArray on which LP pooling is performed
     * @param normType float value indicating norm
     * @param kernelShape the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param padding the padding to be set in each dimension
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the NDArray after applying lp pooling
     */
    public static NDArray lpPool2d(
            NDArray input,
            float normType,
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode) {
        Objects.requireNonNull(kernelShape, "kernelShape cannot be null for lpPool2d");
        Preconditions.checkArgument(
                input.getShape().dimension() == 4,
                "Expect input dimension is 4 but got " + input.getShape().dimension());
        Preconditions.checkArgument(
                kernelShape.dimension() == 2 && stride.dimension() == 2,
                "kernelShape, Stride and Padding dimensions for lpPool2d should be 2");
        return input.getNDArrayInternal().lpPool(normType, kernelShape, stride, padding, ceilMode);
    }

    /**
     * Performs 3-D LP Pooling on the input.
     *
     * @param input the NDArray on which LP pooling is performed
     * @param normType float value indicating norm
     * @param kernelShape the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param padding the padding to be set in each dimension
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the NDArray after applying lp pooling
     */
    public static NDArray lpPool3d(
            NDArray input,
            float normType,
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode) {
        Objects.requireNonNull(kernelShape, "kernelShape cannot be null for lpPool3d");
        Preconditions.checkArgument(
                input.getShape().dimension() == 5,
                "Expect input dimension is 5 but got " + input.getShape().dimension());
        Preconditions.checkArgument(
                kernelShape.dimension() == 3 && stride.dimension() == 3 && padding.dimension() == 3,
                "kernelShape, Stride and Padding dimensions for lpPool3d should be 1");
        return input.getNDArrayInternal().lpPool(normType, kernelShape, stride, padding, ceilMode);
    }

    /**
     * Performs 1-D Global LP Pooling on the input.
     *
     * @param input the NDArray on which LP pooling is performed
     * @param normType float value indicating norm
     * @return the NDArray after applying global lp pooling
     */
    public static NDArray globalLpPool1d(NDArray input, float normType) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 3,
                "Expect input dimension is 3 but got " + input.getShape().dimension());
        return input.getNDArrayInternal().globalLpPool(normType);
    }

    /**
     * Performs 2-D Global LP Pooling on the input.
     *
     * @param input the NDArray on which LP pooling is performed
     * @param normType float value indicating norm
     * @return the NDArray after applying global lp pooling
     */
    public static NDArray globalLpPool2d(NDArray input, float normType) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 4,
                "Expect input dimension is 4 but got " + input.getShape().dimension());
        return input.getNDArrayInternal().globalLpPool(normType);
    }

    /**
     * Performs 3-D Global LP Pooling on the input.
     *
     * @param input the NDArray on which LP pooling is performed
     * @param normType float value indicating norm
     * @return the NDArray after applying global lp pooling
     */
    public static NDArray globalLpPool3d(NDArray input, float normType) {
        Preconditions.checkArgument(
                input.getShape().dimension() == 5,
                "Expect input dimension is 5 but got " + input.getShape().dimension());
        return input.getNDArrayInternal().globalLpPool(normType);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool1d(NDArray, Shape, Shape,
     * Shape, boolean) maxPool1d} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the {@link LambdaBlock} that applies the {@link #maxPool1d(NDArray, Shape, Shape,
     *     Shape, boolean) maxPool1dBlock} activation function
     */
    public static Block maxPool1dBlock(
            Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return LambdaBlock.singleton(
                array -> maxPool1d(array, kernelShape, stride, padding, ceilMode));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool1d(NDArray, Shape, Shape,
     * Shape, boolean) maxPool1dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool1d(NDArray, Shape, Shape,
     *     Shape, boolean) maxPool1dBlock} activation function
     */
    public static Block maxPool1dBlock(Shape kernelShape, Shape stride, Shape padding) {
        return maxPool1dBlock(kernelShape, stride, padding, false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool1d(NDArray, Shape, Shape,
     * Shape, boolean) maxPool1dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool1d(NDArray, Shape, Shape,
     *     Shape, boolean) maxPool1dBlock} activation function
     */
    public static Block maxPool1dBlock(Shape kernelShape, Shape stride) {
        return maxPool1dBlock(kernelShape, stride, new Shape(0), false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool1d(NDArray, Shape, Shape,
     * Shape, boolean) maxPool1dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @return the {@link LambdaBlock} that applies the {@link #maxPool1d(NDArray, Shape, Shape,
     *     Shape, boolean) maxPool1dBlock} activation function
     */
    public static Block maxPool1dBlock(Shape kernelShape) {
        return maxPool1dBlock(kernelShape, kernelShape, new Shape(0), false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool2d(NDArray, Shape, Shape,
     * Shape, boolean) maxPool2dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the {@link LambdaBlock} that applies the {@link #maxPool2d(NDArray, Shape, Shape,
     *     Shape, boolean) maxPool2dBlock} activation function
     */
    public static Block maxPool2dBlock(
            Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return LambdaBlock.singleton(
                array -> maxPool2d(array, kernelShape, stride, padding, ceilMode));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool2d(NDArray, Shape, Shape,
     * Shape, boolean) maxPool2dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool2d(NDArray, Shape, Shape,
     *     Shape, boolean) maxPool2dBlock} activation function
     */
    public static Block maxPool2dBlock(Shape kernelShape, Shape stride, Shape padding) {
        return maxPool2dBlock(kernelShape, stride, padding, false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool2d(NDArray, Shape, Shape,
     * Shape, boolean) maxPool2dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool2d(NDArray, Shape, Shape,
     *     Shape, boolean) maxPool2dBlock} activation function
     */
    public static Block maxPool2dBlock(Shape kernelShape, Shape stride) {
        return maxPool2dBlock(kernelShape, stride, new Shape(0, 0), false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool2d(NDArray, Shape, Shape,
     * Shape, boolean) maxPool2dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @return the {@link LambdaBlock} that applies the {@link #maxPool2d(NDArray, Shape, Shape,
     *     Shape, boolean) maxPool2dBlock} activation function
     */
    public static Block maxPool2dBlock(Shape kernelShape) {
        return maxPool2dBlock(kernelShape, kernelShape, new Shape(0, 0), false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool3d(NDArray, Shape, Shape,
     * Shape, boolean) maxPool3dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the {@link LambdaBlock} that applies the {@link #maxPool3d(NDArray, Shape, Shape,
     *     Shape, boolean) maxPool3dBlock} activation function
     */
    public static Block maxPool3dBlock(
            Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return LambdaBlock.singleton(
                array -> maxPool3d(array, kernelShape, stride, padding, ceilMode));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool3d(NDArray, Shape, Shape,
     * Shape, boolean) maxPool3dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool3d(NDArray, Shape, Shape,
     *     Shape, boolean) maxPool3dBlock} activation function
     */
    public static Block maxPool3dBlock(Shape kernelShape, Shape stride, Shape padding) {
        return maxPool3dBlock(kernelShape, stride, padding, false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool3d(NDArray, Shape, Shape,
     * Shape, boolean) maxPool3dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool3d(NDArray, Shape, Shape,
     *     Shape, boolean) maxPool3dBlock} activation function
     */
    public static Block maxPool3dBlock(Shape kernelShape, Shape stride) {
        return maxPool3dBlock(kernelShape, stride, new Shape(0, 0, 0), false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool3d(NDArray, Shape, Shape,
     * Shape, boolean) maxPool3dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @return the {@link LambdaBlock} that applies the {@link #maxPool3d(NDArray, Shape, Shape,
     *     Shape, boolean) maxPool3dBlock} activation function
     */
    public static Block maxPool3dBlock(Shape kernelShape) {
        return maxPool3dBlock(kernelShape, new Shape(1, 1, 1), new Shape(0, 0, 0), false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalMaxPool1d(NDArray)
     * globalmaxPool1dBlock } pooling function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #globalMaxPool1d(NDArray)
     *     globalmaxPool1dBlock} pooling function
     */
    public static Block globalMaxPool1dBlock() {
        return LambdaBlock.singleton(Pool::globalMaxPool1d);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalMaxPool2d(NDArray)
     * globalmaxPool2dBlock } pooling function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #globalMaxPool2d(NDArray)
     *     globalmaxPool2dBlock} pooling function
     */
    public static Block globalMaxPool2dBlock() {
        return LambdaBlock.singleton(Pool::globalMaxPool2d);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalMaxPool3d(NDArray)
     * globalmaxPool3dBlock } pooling function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #globalMaxPool3d(NDArray)
     *     globalmaxPool3dBlock} pooling function
     */
    public static Block globalMaxPool3dBlock() {
        return LambdaBlock.singleton(Pool::globalMaxPool3d);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool1d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool1dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @param countIncludePad Boolean indicating whether to include padding for calculations
     * @return the {@link LambdaBlock} that applies the {@link #avgPool1d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool1dBlock} activation function
     */
    public static Block avgPool1dBlock(
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        return LambdaBlock.singleton(
                array -> avgPool1d(array, kernelShape, stride, padding, ceilMode, countIncludePad));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool1d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool1dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the {@link LambdaBlock} that applies the {@link #avgPool1d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool1dBlock } activation function
     */
    public static Block avgPool1dBlock(
            Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return avgPool1dBlock(kernelShape, stride, padding, ceilMode, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool1d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool1dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool1d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool1dBlock} activation function
     */
    public static Block avgPool1dBlock(Shape kernelShape, Shape stride, Shape padding) {
        return avgPool1dBlock(kernelShape, stride, padding, false, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool1d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool1dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool1d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool1dBlock} activation function
     */
    public static Block avgPool1dBlock(Shape kernelShape, Shape stride) {
        return avgPool1dBlock(kernelShape, stride, new Shape(0), false, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool1d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool1dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @return the {@link LambdaBlock} that applies the {@link #avgPool1d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool1dBlock} activation function
     */
    public static Block avgPool1dBlock(Shape kernelShape) {
        return avgPool1dBlock(kernelShape, kernelShape, new Shape(0), false, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool2d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool2dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @param countIncludePad Boolean indicating whether to include padding for calculations
     * @return the {@link LambdaBlock} that applies the {@link #avgPool2d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool2dBlock} activation function
     */
    public static Block avgPool2dBlock(
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        return LambdaBlock.singleton(
                array -> avgPool2d(array, kernelShape, stride, padding, ceilMode, countIncludePad));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool2d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool2dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the {@link LambdaBlock} that applies the {@link #avgPool2d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool2dBlock} activation function
     */
    public static Block avgPool2dBlock(
            Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return avgPool2dBlock(kernelShape, stride, padding, ceilMode, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool2d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool2dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool2d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool2dBlock} activation function
     */
    public static Block avgPool2dBlock(Shape kernelShape, Shape stride, Shape padding) {
        return avgPool2dBlock(kernelShape, stride, padding, false, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool2d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool2dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool2d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool2dBlock} activation function
     */
    public static Block avgPool2dBlock(Shape kernelShape, Shape stride) {
        return avgPool2dBlock(kernelShape, stride, new Shape(0, 0), false, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool2d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool2dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @return the {@link LambdaBlock} that applies the {@link #avgPool2d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool2dBlock} activation function
     */
    public static Block avgPool2dBlock(Shape kernelShape) {
        return avgPool2dBlock(kernelShape, kernelShape, new Shape(0, 0), false, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool3d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool3dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @param countIncludePad Boolean indicating whether to include padding for calculations
     * @return the {@link LambdaBlock} that applies the {@link #avgPool3d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool3dBlock} activation function
     */
    public static Block avgPool3dBlock(
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        return LambdaBlock.singleton(
                array -> avgPool3d(array, kernelShape, stride, padding, ceilMode, countIncludePad));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool3d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool3dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the {@link LambdaBlock} that applies the {@link #avgPool3d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool3dBlock} activation function
     */
    public static Block avgPool3dBlock(
            Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return avgPool3dBlock(kernelShape, stride, padding, ceilMode, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool3d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool3dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool3d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool3dBlock} activation function
     */
    public static Block avgPool3dBlock(Shape kernelShape, Shape stride, Shape padding) {
        return avgPool3dBlock(kernelShape, stride, padding, false, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool3d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool3dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool3d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool3dBlock} activation function
     */
    public static Block avgPool3dBlock(Shape kernelShape, Shape stride) {
        return avgPool3dBlock(kernelShape, stride, new Shape(0, 0, 0), false, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool3d(NDArray, Shape, Shape,
     * Shape, boolean, boolean) avgPool3dBlock} pooling function in its forward function.
     *
     * @param kernelShape the shape of the kernel to be used
     * @return the {@link LambdaBlock} that applies the {@link #avgPool3d(NDArray, Shape, Shape,
     *     Shape, boolean, boolean) avgPool3dBlock} activation function
     */
    public static Block avgPool3dBlock(Shape kernelShape) {
        return avgPool3dBlock(kernelShape, kernelShape, new Shape(0, 0, 0), false, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalAvgPool1d(NDArray)
     * globalAvgPool1d } pooling function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #globalAvgPool1d(NDArray)
     *     globalAvgPool1d} pooling function
     */
    public static Block globalAvgPool1dBlock() {
        return LambdaBlock.singleton(Pool::globalAvgPool1d);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalAvgPool2d(NDArray)
     * globalAvgPool2d } pooling function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #globalAvgPool2d(NDArray)
     *     globalAvgPool2d} pooling function
     */
    public static Block globalAvgPool2dBlock() {
        return LambdaBlock.singleton(Pool::globalAvgPool2d);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalAvgPool3d(NDArray)
     * globalAvgPool3d } pooling function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #globalAvgPool3d(NDArray)
     *     globalAvgPool3d} pooling function
     */
    public static Block globalAvgPool3dBlock() {
        return LambdaBlock.singleton(Pool::globalAvgPool3d);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool1d(NDArray, float, Shape, Shape,
     * Shape, boolean) lpPool1dBlock} pooling function in its forward function.
     *
     * @param normType integer indicating pValue
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding padding of pooling layer
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the {@link LambdaBlock} that applies the {@link #lpPool1d(NDArray, float, Shape,
     *     Shape, Shape, boolean) lpPool1dBlock} activation function
     */
    public static Block lpPool1dBlock(
            float normType, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return LambdaBlock.singleton(
                array -> lpPool1d(array, normType, kernelShape, stride, padding, ceilMode));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool1d(NDArray, float, Shape, Shape,
     * Shape, boolean) lpPool1dBlock} pooling function in its forward function.
     *
     * @param normType integer indicating pValue
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding padding of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #lpPool1d(NDArray, float, Shape,
     *     Shape, Shape, boolean) lpPool1dBlock} activation function
     */
    public static Block lpPool1dBlock(
            float normType, Shape kernelShape, Shape stride, Shape padding) {
        return lpPool1dBlock(normType, kernelShape, stride, padding, false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool1d(NDArray, float, Shape, Shape,
     * Shape, boolean) lpPool1dBlock} pooling function in its forward function.
     *
     * @param normType float value indicating norm
     * @param kernelShape the shape of the kernel to be used
     * @return the {@link LambdaBlock} that applies the {@link #lpPool1d(NDArray, float, Shape,
     *     Shape, Shape, boolean) lpPool1dBlock} activation function
     */
    public static Block lpPool1dBlock(float normType, Shape kernelShape) {
        return lpPool1dBlock(normType, kernelShape, new Shape(1), new Shape(0), false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool2d(NDArray, float, Shape, Shape,
     * Shape, boolean) lpPool2dBlock} pooling function in its forward function.
     *
     * @param normType float value indicating norm
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the {@link LambdaBlock} that applies the {@link #lpPool2d(NDArray, float, Shape,
     *     Shape, Shape, boolean) lpPool2dBlock} activation function
     */
    public static Block lpPool2dBlock(
            float normType, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return LambdaBlock.singleton(
                array -> lpPool2d(array, normType, kernelShape, stride, padding, ceilMode));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool2d(NDArray, float, Shape, Shape,
     * Shape, boolean) lpPool2dBlock} pooling function in its forward function.
     *
     * @param normType float value indicating norm
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #lpPool2d(NDArray, float, Shape,
     *     Shape, Shape, boolean) lpPool2dBlock} activation function
     */
    public static Block lpPool2dBlock(
            float normType, Shape kernelShape, Shape stride, Shape padding) {
        return lpPool2dBlock(normType, kernelShape, stride, padding, false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool2d(NDArray, float, Shape, Shape,
     * Shape, boolean) lpPool2dBlock} pooling function in its forward function.
     *
     * @param normType float value indicating norm
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #lpPool2d(NDArray, float, Shape,
     *     Shape, Shape, boolean) lpPool2dBlock} activation function
     */
    public static Block lpPool2dBlock(float normType, Shape kernelShape, Shape stride) {
        return lpPool2dBlock(normType, kernelShape, stride, new Shape(0, 0), false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool2d(NDArray, float, Shape, Shape,
     * Shape, boolean) lpPool2dBlock} pooling function in its forward function.
     *
     * @param normType float value indicating norm
     * @param kernelShape the shape of the kernel to be used
     * @return the {@link LambdaBlock} that applies the {@link #lpPool2d(NDArray, float, Shape,
     *     Shape, Shape, boolean) lpPool2dBlock} activation function
     */
    public static Block lpPool2dBlock(float normType, Shape kernelShape) {
        return lpPool2dBlock(normType, kernelShape, new Shape(1, 1), new Shape(0, 0));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool3d(NDArray, float, Shape, Shape,
     * Shape, boolean) lpPool3dBlock} pooling function in its forward function.
     *
     * @param normType float value indicating norm
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @param ceilMode when true, will use ceil instead of floor in the formula to compute the
     *     output shape. The formula is {@code f(x, k, p, s) = floor((x+2*p-k)/s)+1}.
     * @return the {@link LambdaBlock} that applies the {@link #lpPool3d(NDArray, float, Shape,
     *     Shape, Shape, boolean) lpPool3dBlock} activation function
     */
    public static Block lpPool3dBlock(
            float normType, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return LambdaBlock.singleton(
                array -> lpPool3d(array, normType, kernelShape, stride, padding, ceilMode));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool3d(NDArray, float, Shape, Shape,
     * Shape, boolean) lpPool3dBlock} pooling function in its forward function.
     *
     * @param normType float value indicating norm
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @param padding pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #lpPool3d(NDArray, float, Shape,
     *     Shape, Shape, boolean) lpPool3dBlock} activation function
     */
    public static Block lpPool3dBlock(
            float normType, Shape kernelShape, Shape stride, Shape padding) {
        return lpPool3dBlock(normType, kernelShape, stride, padding, false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool3d(NDArray, float, Shape, Shape,
     * Shape, boolean) LpPoo3D} pooling function in its forward function.
     *
     * @param normType float value indicating norm
     * @param kernelShape the shape of the kernel to be used
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #lpPool3d(NDArray, float, Shape,
     *     Shape, Shape, boolean) lpPool3dBlock} activation function
     */
    public static Block lpPool3dBlock(float normType, Shape kernelShape, Shape stride) {
        return lpPool3dBlock(normType, kernelShape, stride, new Shape(0, 0, 0), false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool3d(NDArray, float, Shape, Shape,
     * Shape, boolean) lpPool3dBlock} pooling function in its forward function.
     *
     * @param normType float value indicating norm
     * @param kernelShape the shape of the kernel to be used
     * @return the {@link LambdaBlock} that applies the {@link #lpPool3d(NDArray, float, Shape,
     *     Shape, Shape, boolean) lpPool3dBlock} activation function
     */
    public static Block lpPool3dBlock(float normType, Shape kernelShape) {
        return lpPool3dBlock(normType, kernelShape, kernelShape, new Shape(0, 0, 0), false);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalLpPool1d(NDArray, float)
     * globalLpPool1d } pooling function.
     *
     * @param normType float value indicating norm
     * @return the {@link LambdaBlock} that applies the {@link #globalLpPool1d(NDArray, float)
     *     globalLpPool1d} pooling function
     */
    public static Block globalLpPool1dBlock(float normType) {
        return LambdaBlock.singleton(array -> globalLpPool1d(array, normType));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalLpPool2d(NDArray, float)
     * globalLpPool2d } pooling function.
     *
     * @param normType float value indicating norm
     * @return the {@link LambdaBlock} that applies the {@link #globalLpPool2d(NDArray, float)
     *     globalLpPool2d} pooling function
     */
    public static Block globalLpPool2dBlock(float normType) {
        return LambdaBlock.singleton(array -> globalLpPool2d(array, normType));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalLpPool3d(NDArray, float)
     * globalLpPool3d } pooling function.
     *
     * @param normType float value indicating norm
     * @return the {@link LambdaBlock} that applies the {@link #globalLpPool3d(NDArray, float)
     *     globalLpPool3d} pooling function
     */
    public static Block globalLpPool3dBlock(float normType) {
        return LambdaBlock.singleton(array -> globalLpPool3d(array, normType));
    }
}
