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
package software.amazon.ai.modality.cv.util;

import software.amazon.ai.ndarray.NDArray;

/**
 * {@code NDImageUtils} is an image processing utility that can load, reshape and convert images
 * using {@link NDArray} images.
 */
public final class NDImageUtils {
    private NDImageUtils() {}

    public static NDArray resize(NDArray image, int[] size) {
        return image.getNDArrayInternal().resize(size);
    }

    /**
     * Normalize a NDArray of shape (C x H x W) or (N x C x H x W) with mean and standard deviation.
     *
     * <p>Given mean `(m1, ..., mn)` and std `(s\ :sub:`1`\ , ..., s\ :sub:`n`)` for `n` channels,
     * this transform normalizes each channel of the input tensor with: output[i] = (input[i] - m\
     * :sub:`i`\ ) / s\ :sub:`i`
     *
     * @param input the input image NDArray
     * @param mean mean value for each channel
     * @param std standard deviation for each channel
     * @return the result of normalization
     */
    public static NDArray normalize(NDArray input, float[] mean, float[] std) {
        return input.getNDArrayInternal().normalize(mean, std);
    }

    public static NDArray toTensor(NDArray image) {
        return image.getNDArrayInternal().toTensor();
    }

    public static NDArray crop(NDArray image, int x, int y, int width, int height) {
        return image.getNDArrayInternal().crop(x, y, width, height);
    }

    public enum Flag {
        GRAYSCALE,
        COLOR
    }
}
