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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;

/** Utility class that provides {@code Block} and methods for different pooling functions. */
public final class Pool {

    private Pool() {}

    /**
     * Performs Max Pooling on the input.
     *
     * @param data the NDArray on which max pooling is performed
     * @param kernel the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param pad the padding to be set in each dimension
     * @param poolingConvention the pooling convention to be used
     * @return the NDArray after applying max pooling
     */
    private static NDArray maxPool(
            NDArray data,
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention) {
        return data.getNDArrayInternal().maxPool(kernel, stride, pad, poolingConvention);
    }
    /**
     * Performs Max Pooling on the input NDList.
     *
     * @param list the NDList on which max pooling is performed
     * @param kernel the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param pad the padding to be set in each dimension
     * @param poolingConvention the pooling convention to be used
     * @return the NDList after applying max pooling
     */
    private static NDList maxPool(
            NDList list,
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention) {

        return new NDList(maxPool(list.singletonOrThrow(), kernel, stride, pad, poolingConvention));
    }

    /**
     * Performs Global Max Pooling on the input.
     *
     * @param data the NDArray on which max pooling is performed
     * @return the NDArray after applying global max pooling
     */
    private static NDArray globalMaxPool(NDArray data) {
        return data.getNDArrayInternal().globalMaxPool();
    }

    /**
     * Performs Avg Pooling on the input.
     *
     * @param data the NDArray on which average pooling is performed
     * @param kernel the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param pad the padding to be set in each dimension
     * @param poolingConvention the pooling convention to be used
     * @param countIncludePad whether to include padding for calculations
     * @return the NDArray after applying avg pooling
     */
    private static NDArray avgPool(
            NDArray data,
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            boolean countIncludePad) {
        return data.getNDArrayInternal()
                .avgPool(kernel, stride, pad, poolingConvention, countIncludePad);
    }

    /**
     * Performs Avg Pooling on the input.
     *
     * @param data the NDArray on which average pooling is performed
     * @param kernel the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param pad the padding to be set in each dimension
     * @return the NDArray after applying avg pooling
     */
    public static NDArray avgPool(NDArray data, Shape kernel, Shape stride, Shape pad) {
        return avgPool(data, kernel, stride, pad, PoolingConvention.VALID, true);
    }

    /**
     * Performs Avg Pooling on the input.
     *
     * @param list the NDList on which average pooling is performed
     * @param kernel the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param pad the padding to be set in each dimension
     * @param poolingConvention the pooling convention to be used
     * @param countIncludePad whether to include padding for calculations
     * @return the NDList after applying avg pooling
     */
    private static NDList avgPool(
            NDList list,
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            boolean countIncludePad) {
        return new NDList(
                avgPool(
                        list.singletonOrThrow(),
                        kernel,
                        stride,
                        pad,
                        poolingConvention,
                        countIncludePad));
    }

    /**
     * Performs Global Avg Pooling on the input.
     *
     * @param data the NDArray on which average pooling is performed
     * @return the NDArray after applying global avg pooling
     */
    private static NDArray globalAvgPool(NDArray data) {
        return data.getNDArrayInternal().globalAvgPool();
    }

    /**
     * Performs LP Pooling on the input.
     *
     * @param data the NDArray on which LP pooling is performed
     * @param kernel the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param pad the padding to be set in each dimension
     * @param poolingConvention the pooling convention to be used
     * @param pValue the power of the pooling
     * @return the NDArray after applying lp pooling
     */
    private static NDArray lpPool(
            NDArray data,
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            int pValue) {
        return data.getNDArrayInternal().lpPool(kernel, stride, pad, poolingConvention, pValue);
    }

    /**
     * Performs LP Pooling on the input.
     *
     * @param list the NDList on which LP pooling is performed
     * @param kernel the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param pad the padding to be set in each dimension
     * @param poolingConvention the pooling convention to be used
     * @param pValue the power of the pooling
     * @return the NDList after applying lp pooling
     */
    private static NDList lpPool(
            NDList list,
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            int pValue) {
        return new NDList(
                lpPool(list.singletonOrThrow(), kernel, stride, pad, poolingConvention, pValue));
    }

    /**
     * Performs Global LP Pooling on the input.
     *
     * @param data the NDArray on which LP pooling is performed
     * @param pValue the power of the pooling
     * @return the NDArray after applying global lp pooling
     */
    private static NDArray globalLpPool(NDArray data, int pValue) {
        return data.getNDArrayInternal().globalLpPool(pValue);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention) maxPool1D} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param poolingConvention PoolingConvention (VALID vs FULL)
     * @return the {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention) maxPool1DBlock} activation function
     */
    public static Block maxPool1DBlock(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        if (kernel == null) {
            throw new IllegalArgumentException("Kernel cannot be null for maxPool1DBlock Block");
        }
        if (kernel.dimension() != 1 || stride.dimension() != 1 || pad.dimension() != 1) {
            throw new IllegalArgumentException(
                    "Kernel , Stride and Pad dimensions for maxPool1DBlock layer should be 1");
        }
        return new LambdaBlock(ndList -> maxPool(ndList, kernel, stride, pad, poolingConvention));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention) maxPool1DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention) maxPool1DBlock} activation function
     */
    public static Block maxPool1DBlock(Shape kernel, Shape stride, Shape pad) {
        return maxPool1DBlock(kernel, stride, pad, PoolingConvention.VALID);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention) maxPool1DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention) maxPool1DBlock} activation function
     */
    public static Block maxPool1DBlock(Shape kernel, Shape stride) {
        return maxPool1DBlock(kernel, stride, new Shape(0), PoolingConvention.VALID);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention) maxPool1DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention) maxPool1DBlock} activation function
     */
    public static Block maxPool1DBlock(Shape kernel) {
        return maxPool1DBlock(kernel, kernel, new Shape(0), PoolingConvention.VALID);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention) maxPool2DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param poolingConvention PoolingConvention (VALID vs FULL)
     * @return the {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention) maxPool2DBlock} activation function
     */
    public static Block maxPool2DBlock(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        if (kernel == null) {
            throw new IllegalArgumentException("Kernel cannot be null for maxPool2DBlock Block");
        }
        if (kernel.dimension() != 2 || stride.dimension() != 2 || pad.dimension() != 2) {
            throw new IllegalArgumentException(
                    "Kernel , Stride and Pad dimensions for maxPool2DBlock layer should be 2");
        }
        return new LambdaBlock(ndList -> maxPool(ndList, kernel, stride, pad, poolingConvention));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention) maxPool2DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention) maxPool2DBlock} activation function
     */
    public static Block maxPool2DBlock(Shape kernel, Shape stride, Shape pad) {
        return maxPool2DBlock(kernel, stride, pad, PoolingConvention.VALID);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention) maxPool2DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention) maxPool2DBlock} activation function
     */
    public static Block maxPool2DBlock(Shape kernel, Shape stride) {
        return maxPool2DBlock(kernel, stride, new Shape(0, 0), PoolingConvention.VALID);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention) maxPool2DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention) maxPool2DBlock} activation function
     */
    public static Block maxPool2DBlock(Shape kernel) {
        return maxPool2DBlock(kernel, kernel, new Shape(0, 0), PoolingConvention.VALID);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention) maxPool3DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param poolingConvention PoolingConvention (VALID vs FULL)
     * @return the {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention) maxPool3DBlock} activation function
     */
    public static Block maxPool3DBlock(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        if (kernel == null) {
            throw new IllegalArgumentException("Kernel cannot be null for maxPool3DBlock Block");
        }
        if (kernel.dimension() != 3 || stride.dimension() != 3 || pad.dimension() != 3) {
            throw new IllegalArgumentException(
                    "Kernel , Stride and Pad dimensions for maxPool3DBlock layer should be 3");
        }
        return new LambdaBlock(ndList -> maxPool(ndList, kernel, stride, pad, poolingConvention));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention) maxPool3DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention) maxPool3DBlock} activation function
     */
    public static Block maxPool3DBlock(Shape kernel, Shape stride, Shape pad) {
        return maxPool3DBlock(kernel, stride, pad, PoolingConvention.VALID);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention) maxPool3DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention) maxPool3DBlock} activation function
     */
    public static Block maxPool3DBlock(Shape kernel, Shape stride) {
        return maxPool3DBlock(kernel, stride, new Shape(0, 0, 0), PoolingConvention.VALID);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention) maxPool3DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #maxPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention) maxPool3DBlock} activation function
     */
    public static Block maxPool3DBlock(Shape kernel) {
        return maxPool3DBlock(kernel, kernel, new Shape(0, 0, 0), PoolingConvention.VALID);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalMaxPool(NDArray)
     * globalmaxPool1DBlock } pooling function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #globalMaxPool(NDArray)
     *     globalmaxPool1DBlock} pooling function
     */
    public static Block globalMaxPool1DBlock() {
        return new LambdaBlock(ndList -> new NDList(globalMaxPool(ndList.singletonOrThrow())));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalMaxPool(NDArray)
     * globalmaxPool2DBlock } pooling function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #globalMaxPool(NDArray)
     *     globalmaxPool2DBlock} pooling function
     */
    public static Block globalMaxPool2DBlock() {
        return new LambdaBlock(ndList -> new NDList(globalMaxPool(ndList.singletonOrThrow())));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalMaxPool(NDArray)
     * globalmaxPool3DBlock } pooling function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #globalMaxPool(NDArray)
     *     globalmaxPool3DBlock} pooling function
     */
    public static Block globalMaxPool3DBlock() {
        return new LambdaBlock(ndList -> new NDList(globalMaxPool(ndList.singletonOrThrow())));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool1DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param poolingConvention PoolingConvention (VALID vs FULL)
     * @param countIncludePad Boolean indicating whether to include padding for calculations
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool1DBlock} activation function
     */
    public static Block avgPool1DBlock(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            boolean countIncludePad) {
        if (kernel == null) {
            throw new IllegalArgumentException("Kernel cannot be null for avgPool1DBlock Block");
        }
        if (kernel.dimension() != 1 || stride.dimension() != 1 || pad.dimension() != 1) {
            throw new IllegalArgumentException(
                    "Kernel , Stride and Pad dimensions for avgPool1DBlock layer should be 1");
        }
        return new LambdaBlock(
                ndList -> avgPool(ndList, kernel, stride, pad, poolingConvention, countIncludePad));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool1DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param poolingConvention PoolingConvention (VALID vs FULL)
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool1DBlock } activation function
     */
    public static Block avgPool1DBlock(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        return avgPool1DBlock(kernel, stride, pad, poolingConvention, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool1DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool1DBlock} activation function
     */
    public static Block avgPool1DBlock(Shape kernel, Shape stride, Shape pad) {
        return avgPool1DBlock(kernel, stride, pad, PoolingConvention.VALID, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool1DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool1DBlock} activation function
     */
    public static Block avgPool1DBlock(Shape kernel, Shape stride) {
        return avgPool1DBlock(kernel, stride, new Shape(0), PoolingConvention.VALID, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool1DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool1DBlock} activation function
     */
    public static Block avgPool1DBlock(Shape kernel) {
        return avgPool1DBlock(kernel, kernel, new Shape(0), PoolingConvention.VALID, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool2DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param poolingConvention PoolingConvention (VALID vs FULL)
     * @param countIncludePad Boolean indicating whether to include padding for calculations
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool2DBlock} activation function
     */
    public static Block avgPool2DBlock(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            boolean countIncludePad) {
        if (kernel == null) {
            throw new IllegalArgumentException("Kernel cannot be null for avgPool3DBlock Block");
        }
        if (kernel.dimension() != 2 || stride.dimension() != 2 || pad.dimension() != 2) {
            throw new IllegalArgumentException(
                    "Kernel , Stride and Pad dimensions for avgPool2DBlock layer should be 2");
        }
        return new LambdaBlock(
                ndList -> avgPool(ndList, kernel, stride, pad, poolingConvention, countIncludePad));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool2DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param poolingConvention PoolingConvention (VALID vs FULL)
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool2DBlock} activation function
     */
    public static Block avgPool2DBlock(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        return avgPool2DBlock(kernel, stride, pad, poolingConvention, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool2DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool2DBlock} activation function
     */
    public static Block avgPool2DBlock(Shape kernel, Shape stride, Shape pad) {
        return avgPool2DBlock(kernel, stride, pad, PoolingConvention.VALID, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool2DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool2DBlock} activation function
     */
    public static Block avgPool2DBlock(Shape kernel, Shape stride) {
        return avgPool2DBlock(kernel, stride, new Shape(0, 0), PoolingConvention.VALID, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool2DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool2DBlock} activation function
     */
    public static Block avgPool2DBlock(Shape kernel) {
        return avgPool2DBlock(kernel, kernel, new Shape(0, 0), PoolingConvention.VALID, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool3DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param poolingConvention PoolingConvention (VALID vs FULL)
     * @param countIncludePad Boolean indicating whether to include padding for calculations
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool3DBlock} activation function
     */
    public static Block avgPool3DBlock(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            boolean countIncludePad) {
        if (kernel == null) {
            throw new IllegalArgumentException("Kernel cannot be null for avgPool3DBlock Block");
        }
        if (kernel.dimension() != 3 || stride.dimension() != 3 || pad.dimension() != 3) {
            throw new IllegalArgumentException(
                    "Kernel , Stride and Pad dimensions for avgPool3DBlock layer should be 3");
        }
        return new LambdaBlock(
                ndList -> avgPool(ndList, kernel, stride, pad, poolingConvention, countIncludePad));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool3DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param poolingConvention PoolingConvention (VALID vs FULL)
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool3DBlock} activation function
     */
    public static Block avgPool3DBlock(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention) {
        return avgPool3DBlock(kernel, stride, pad, poolingConvention, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool3DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool3DBlock} activation function
     */
    public static Block avgPool3DBlock(Shape kernel, Shape stride, Shape pad) {
        return avgPool3DBlock(kernel, stride, pad, PoolingConvention.VALID, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool3DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool3DBlock} activation function
     */
    public static Block avgPool3DBlock(Shape kernel, Shape stride) {
        return avgPool3DBlock(kernel, stride, new Shape(0, 0, 0), PoolingConvention.VALID, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, boolean) avgPool3DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @return the {@link LambdaBlock} that applies the {@link #avgPool(NDArray, Shape, Shape,
     *     Shape, PoolingConvention, boolean) avgPool3DBlock} activation function
     */
    public static Block avgPool3DBlock(Shape kernel) {
        return avgPool3DBlock(kernel, kernel, new Shape(0, 0, 0), PoolingConvention.VALID, true);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalAvgPool(NDArray) globalAvgPool1D
     * } pooling function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #globalAvgPool(NDArray)
     *     globalAvgPool1D} pooling function
     */
    public static Block globalAvgPool1DBlock() {
        return new LambdaBlock(ndList -> new NDList(globalAvgPool(ndList.singletonOrThrow())));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalAvgPool(NDArray) globalAvgPool2D
     * } pooling function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #globalAvgPool(NDArray)
     *     globalAvgPool2D} pooling function
     */
    public static Block globalAvgPool2DBlock() {
        return new LambdaBlock(ndList -> new NDList(globalAvgPool(ndList.singletonOrThrow())));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalAvgPool(NDArray) globalAvgPool3D
     * } pooling function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #globalAvgPool(NDArray)
     *     globalAvgPool3D} pooling function
     */
    public static Block globalAvgPool3DBlock() {
        return new LambdaBlock(ndList -> new NDList(globalAvgPool(ndList.singletonOrThrow())));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, int) lpPool1DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param poolingConvention PoolingConvention (VALID vs FULL)
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     *     PoolingConvention, int) lpPool1DBlock} activation function
     */
    public static Block lpPool1DBlock(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            int pValue) {
        if (kernel == null) {
            throw new IllegalArgumentException("Kernel cannot be null for lpPool1D Block");
        }

        if (kernel.dimension() != 1 || stride.dimension() != 1 || pad.dimension() != 1) {
            throw new IllegalArgumentException(
                    "Kernel , Stride and Pad dimensions for lpPool1D layer should be 1");
        }
        return new LambdaBlock(
                ndList -> lpPool(ndList, kernel, stride, pad, poolingConvention, pValue));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, int) lpPool1DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     *     PoolingConvention, int) lpPool1DBlock} activation function
     */
    public static Block lpPool1DBlock(Shape kernel, Shape stride, Shape pad, int pValue) {
        return lpPool1DBlock(kernel, stride, pad, PoolingConvention.VALID, pValue);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, int) lpPool1DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     *     PoolingConvention, int) lpPool1DBlock} activation function
     */
    public static Block lpPool1DBlock(Shape kernel, Shape stride, int pValue) {
        return lpPool1DBlock(kernel, stride, new Shape(0), PoolingConvention.VALID, pValue);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, int) lpPool1DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     *     PoolingConvention, int) lpPool1DBlock} activation function
     */
    public static Block lpPool1DBlock(Shape kernel, int pValue) {
        return lpPool1DBlock(kernel, kernel, new Shape(0), PoolingConvention.VALID, pValue);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, int) lpPool2DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param poolingConvention PoolingConvention (VALID vs FULL)
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     *     PoolingConvention, int) lpPool2DBlock} activation function
     */
    public static Block lpPool2DBlock(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            int pValue) {
        if (kernel == null) {
            throw new IllegalArgumentException("Kernel cannot be null for lpPool2D Block");
        }

        if (kernel.dimension() != 2 || stride.dimension() != 2 || pad.dimension() != 2) {
            throw new IllegalArgumentException(
                    "Kernel , Stride and Pad dimensions for lpPool2D layer should be 2");
        }
        return new LambdaBlock(
                ndList -> lpPool(ndList, kernel, stride, pad, poolingConvention, pValue));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, int) lpPool2DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     *     PoolingConvention, int) lpPool2DBlock} activation function
     */
    public static Block lpPool2DBlock(Shape kernel, Shape stride, Shape pad, int pValue) {
        return lpPool2DBlock(kernel, stride, pad, PoolingConvention.VALID, pValue);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, int) lpPool2DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     *     PoolingConvention, int) lpPool2DBlock} activation function
     */
    public static Block lpPool2DBlock(Shape kernel, Shape stride, int pValue) {
        return lpPool2DBlock(kernel, stride, new Shape(0, 0), PoolingConvention.VALID, pValue);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, int) lpPool2DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     *     PoolingConvention, int) lpPool2DBlock} activation function
     */
    public static Block lpPool2DBlock(Shape kernel, int pValue) {
        return lpPool2DBlock(kernel, kernel, new Shape(0, 0), PoolingConvention.VALID, pValue);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, int) lpPool3DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param poolingConvention PoolingConvention (VALID vs FULL)
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     *     PoolingConvention, int) lpPool3DBlock} activation function
     */
    public static Block lpPool3DBlock(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            int pValue) {
        if (kernel == null) {
            throw new IllegalArgumentException("Kernel cannot be null for lpPool3D Block");
        }

        if (kernel.dimension() != 3 || stride.dimension() != 3 || pad.dimension() != 3) {
            throw new IllegalArgumentException(
                    "Kernel , Stride and Pad dimensions for lpPool3D layer should be 3");
        }
        return new LambdaBlock(
                ndList -> lpPool(ndList, kernel, stride, pad, poolingConvention, pValue));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, int) lpPool3DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pad pad of the pooling layer
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     *     PoolingConvention, int) lpPool3DBlock} activation function
     */
    public static Block lpPool3DBlock(Shape kernel, Shape stride, Shape pad, int pValue) {
        return lpPool3DBlock(kernel, stride, pad, PoolingConvention.VALID, pValue);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, int) LpPoo3D} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param stride stride of pooling layer
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     *     PoolingConvention, int) lpPool3DBlock} activation function
     */
    public static Block lpPool3DBlock(Shape kernel, Shape stride, int pValue) {
        return lpPool3DBlock(kernel, stride, new Shape(0, 0, 0), PoolingConvention.VALID, pValue);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     * PoolingConvention, int) lpPool3DBlock} pooling function in its forward function.
     *
     * @param kernel kernel of pooling layer
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #lpPool(NDArray, Shape, Shape, Shape,
     *     PoolingConvention, int) lpPool3DBlock} activation function
     */
    public static Block lpPool3DBlock(Shape kernel, int pValue) {
        return lpPool3DBlock(kernel, kernel, new Shape(0, 0, 0), PoolingConvention.VALID, pValue);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalLpPool(NDArray, int)
     * globalLpPool1D } pooling function.
     *
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #globalLpPool(NDArray, int)
     *     globalLpPool1D} pooling function
     */
    public static Block globalLpPool1DBlock(int pValue) {
        return new LambdaBlock(
                ndList -> new NDList(globalLpPool(ndList.singletonOrThrow(), pValue)));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalLpPool(NDArray, int)
     * globalLpPool2D } pooling function.
     *
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #globalLpPool(NDArray, int)
     *     globalLpPool2D} pooling function
     */
    public static Block globalLpPool2DBlock(int pValue) {
        return new LambdaBlock(
                ndList -> new NDList(globalLpPool(ndList.singletonOrThrow(), pValue)));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #globalLpPool(NDArray, int)
     * globalLpPool3D } pooling function.
     *
     * @param pValue integer indicating pValue
     * @return the {@link LambdaBlock} that applies the {@link #globalLpPool(NDArray, int)
     *     globalLpPool3D} pooling function
     */
    public static Block globalLpPool3DBlock(int pValue) {
        return new LambdaBlock(
                ndList -> new NDList(globalLpPool(ndList.singletonOrThrow(), pValue)));
    }
}
