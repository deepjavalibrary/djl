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
    public static NDArray maxPool(
            NDArray data,
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention) {
        return data.getNDArrayInternal().maxPool(kernel, stride, pad, poolingConvention);
    }

    /**
     * Performs Max Pooling on the input.
     *
     * @param data the NDArray on which max pooling is performed
     * @param kernel the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param pad the padding to be set in each dimension
     * @return the NDArray after applying max pooling
     */
    public static NDArray maxPool(NDArray data, Shape kernel, Shape stride, Shape pad) {
        return maxPool(data, kernel, stride, pad, PoolingConvention.VALID);
    }

    /**
     * Performs Global Max Pooling on the input.
     *
     * @param data the NDArray on which max pooling is performed
     * @return the NDArray after applying global max pooling
     */
    public static NDArray globalMaxPool(NDArray data) {
        return data.getNDArrayInternal().globalMaxPool();
    }

    /**
     * Performs Sum Pooling on the input.
     *
     * @param data the NDArray on which sum pooling is performed
     * @param kernel the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param pad the padding to be set in each dimension
     * @param poolingConvention the pooling convention to be used
     * @return the NDArray after applying sum pooling
     */
    public static NDArray sumPool(
            NDArray data,
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention) {
        return data.getNDArrayInternal().sumPool(kernel, stride, pad, poolingConvention);
    }

    /**
     * Performs Sum Pooling on the input.
     *
     * @param data the NDArray on which sum pooling is performed
     * @param kernel the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param pad the padding to be set in each dimension
     * @return the NDArray after applying sum pooling
     */
    public static NDArray sumPool(NDArray data, Shape kernel, Shape stride, Shape pad) {
        return sumPool(data, kernel, stride, pad, PoolingConvention.VALID);
    }

    /**
     * Performs Global Sum Pooling on the input.
     *
     * @param data the NDArray on which sum pooling is performed
     * @return the NDArray after applying global sum pooling
     */
    public static NDArray globalSumPool(NDArray data) {
        return data.getNDArrayInternal().globalSumPool();
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
    public static NDArray avgPool(
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
     * Performs Global Avg Pooling on the input.
     *
     * @param data the NDArray on which average pooling is performed
     * @return the NDArray after applying global avg pooling
     */
    public static NDArray globalAvgPool(NDArray data) {
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
    public static NDArray lpPool(
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
     * @param data the NDArray on which LP pooling is performed
     * @param kernel the shape of the kernel to be used
     * @param stride the stride to be used for each dimension
     * @param pad the padding to be set in each dimension
     * @param pValue the power of the pooling
     * @return the NDArray after applying lp pooling
     */
    public static NDArray lpPool(NDArray data, Shape kernel, Shape stride, Shape pad, int pValue) {
        return data.getNDArrayInternal()
                .lpPool(kernel, stride, pad, PoolingConvention.VALID, pValue);
    }

    /**
     * Performs Global LP Pooling on the input.
     *
     * @param data the NDArray on which LP pooling is performed
     * @param pValue the power of the pooling
     * @return the NDArray after applying global lp pooling
     */
    public static NDArray globalLpPool(NDArray data, int pValue) {
        return data.getNDArrayInternal().globalLpPool(pValue);
    }
}
