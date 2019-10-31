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
package ai.djl.modality.cv.util;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;

/**
 * {@code NDImageUtils} is an image processing utility that can load, reshape, and convert images
 * using {@link NDArray} images.
 */
public final class NDImageUtils {

    private NDImageUtils() {}

    public static NDArray resize(NDArray image, int size) {
        return image.getNDArrayInternal().resize(size, size);
    }

    public static NDArray resize(NDArray image, int width, int height) {
        return image.getNDArrayInternal().resize(width, height);
    }

    public static NDArray normalize(NDArray input, float mean, float std) {
        return normalize(input, new float[] {mean, mean, mean}, new float[] {std, std, std});
    }

    /**
     * Normalizes a NDArray of shape (C x H x W) or (N x C x H x W) with mean and standard
     * deviation.
     *
     * <p>Given mean `(m1, ..., mn)` and std `(s\ :sub:`1`\ , ..., s\ :sub:`n`)` for `n` channels,
     * this transform normalizes each channel of the input tensor with: output[i] = (input[i] - m\
     * :sub:`i`\ ) / s\ :sub:`i`
     *
     * @param input the input image NDArray
     * @param mean the mean value for each channel
     * @param std the standard deviation for each channel
     * @return the result of normalization
     */
    public static NDArray normalize(NDArray input, float[] mean, float[] std) {
        return input.getNDArrayInternal().normalize(mean, std);
    }

    public static NDArray toTensor(NDArray image) {
        return image.getNDArrayInternal().toTensor();
    }

    public static NDArray centerCrop(NDArray image) {
        Shape shape = image.getShape();
        int w = (int) shape.get(1);
        int h = (int) shape.get(0);

        if (w == h) {
            return image;
        }

        if (w > h) {
            return centerCrop(image, h, h);
        }

        return centerCrop(image, w, w);
    }

    public static NDArray centerCrop(NDArray image, int width, int height) {
        Shape shape = image.getShape();
        int w = (int) shape.get(1);
        int h = (int) shape.get(0);

        int x;
        int y;
        int dw = (w - width) / 2;
        int dh = (h - height) / 2;
        if (dw > 0) {
            x = dw;
            w = width;
        } else {
            x = 0;
        }
        if (dh > 0) {
            y = dh;
            h = height;
        } else {
            y = 0;
        }

        return crop(image, x, y, w, h);
    }

    public static NDArray crop(NDArray image, int x, int y, int width, int height) {
        return image.getNDArrayInternal().crop(x, y, width, height);
    }

    public enum Flag {
        GRAYSCALE,
        COLOR
    }
}
