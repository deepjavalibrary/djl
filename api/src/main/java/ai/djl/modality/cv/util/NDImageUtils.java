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

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.RandomUtils;

/**
 * {@code NDImageUtils} is an image processing utility to load, reshape, and convert images using
 * {@link NDArray} images.
 */
public final class NDImageUtils {

    private NDImageUtils() {}

    /**
     * Resizes an image to the given width and height.
     *
     * @param image the image to resize
     * @param size the desired size
     * @return the resized NDList
     */
    public static NDArray resize(NDArray image, int size) {
        return resize(image, size, size, Image.Interpolation.BILINEAR);
    }

    /**
     * Resizes an image to the given width and height.
     *
     * @param image the image to resize
     * @param width the desired width
     * @param height the desired height
     * @return the resized NDList
     */
    public static NDArray resize(NDArray image, int width, int height) {
        return resize(image, width, height, Image.Interpolation.BILINEAR);
    }

    /**
     * Resizes an image to the given width and height with given interpolation.
     *
     * @param image the image to resize
     * @param width the desired width
     * @param height the desired height
     * @param interpolation the desired interpolation
     * @return the resized NDList
     */
    public static NDArray resize(
            NDArray image, int width, int height, Image.Interpolation interpolation) {
        return image.getNDArrayInternal().resize(width, height, interpolation.ordinal());
    }

    /**
     * Rotate an image NDArray counter-clockwise 90 degree.
     *
     * @param image the image to rotate
     * @param times the image to rotate
     * @return the rotated Image
     */
    public static NDArray rotate90(NDArray image, int times) {
        Shape shape = image.getShape();
        int batchDim = shape.dimension() == 4 ? 1 : 0;
        if (isCHW(shape)) {
            return image.rotate90(times, new int[] {1 + batchDim, 2 + batchDim});
        } else {
            return image.rotate90(times, new int[] {batchDim, 1 + batchDim});
        }
    }

    /**
     * Normalizes an image NDArray of shape CHW or NCHW with a single mean and standard deviation to
     * apply to all channels. TensorFlow enforce HWC instead.
     *
     * @param input the image to normalize
     * @param mean the mean to normalize with (for all channels)
     * @param std the standard deviation to normalize with (for all channels)
     * @return the normalized NDArray
     * @see NDImageUtils#normalize(NDArray, float[], float[])
     */
    public static NDArray normalize(NDArray input, float mean, float std) {
        return normalize(input, new float[] {mean, mean, mean}, new float[] {std, std, std});
    }

    /**
     * Normalizes an image NDArray of shape CHW or NCHW with mean and standard deviation. TensorFlow
     * enforce HWC instead.
     *
     * <p>Given mean {@code (m1, ..., mn)} and standard deviation {@code (s1, ..., sn} for {@code n}
     * channels, this transform normalizes each channel of the input tensor with: {@code output[i] =
     * (input[i] - m1) / (s1)}.
     *
     * @param input the image to normalize
     * @param mean the mean to normalize with for each channel
     * @param std the standard deviation to normalize with for each channel
     * @return the normalized NDArray
     */
    public static NDArray normalize(NDArray input, float[] mean, float[] std) {
        boolean chw = isCHW(input.getShape());
        boolean tf = "TensorFlow".equals(input.getManager().getEngine().getEngineName());
        if ((chw && tf) || (!chw && !tf)) {
            throw new IllegalArgumentException(
                    "normalize requires CHW format. TensorFlow requires HWC");
        }
        return input.getNDArrayInternal().normalize(mean, std);
    }

    /**
     * Converts an image NDArray from preprocessing format to Neural Network format.
     *
     * <p>Converts an image NDArray of shape HWC in the range {@code [0, 255]} to a {@link
     * ai.djl.ndarray.types.DataType#FLOAT32} tensor NDArray of shape CHW in the range {@code [0,
     * 1]}.
     *
     * @param image the image to convert
     * @return the converted image
     */
    public static NDArray toTensor(NDArray image) {
        return image.getNDArrayInternal().toTensor();
    }

    /**
     * Crops an image to a square of size {@code min(width, height)}.
     *
     * @param image the image to crop
     * @return the cropped image
     * @see NDImageUtils#centerCrop(NDArray, int, int)
     */
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

    /**
     * Crops an image to a given width and height from the center of the image.
     *
     * @param image the image to crop
     * @param width the desired width of the cropped image
     * @param height the desired height of the cropped image
     * @return the cropped image
     */
    public static NDArray centerCrop(NDArray image, int width, int height) {
        Shape shape = image.getShape();
        if (isCHW(image.getShape()) || shape.dimension() == 4) {
            throw new IllegalArgumentException("CenterCrop only support for HWC image format");
        }
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

    /**
     * Crops an image with a given location and size.
     *
     * @param image the image to crop
     * @param x the x coordinate of the top-left corner of the crop
     * @param y the y coordinate of the top-left corner of the crop
     * @param width the width of the cropped image
     * @param height the height of the cropped image
     * @return the cropped image
     */
    public static NDArray crop(NDArray image, int x, int y, int width, int height) {
        return image.getNDArrayInternal().crop(x, y, width, height);
    }

    /**
     * Randomly flip the input image left to right with a probability of 0.5.
     *
     * @param image the image with HWC format
     * @return the flipped image
     */
    public static NDArray randomFlipLeftRight(NDArray image) {
        return image.getNDArrayInternal().randomFlipLeftRight();
    }

    /**
     * Randomly flip the input image top to bottom with a probability of 0.5.
     *
     * @param image the image with HWC format
     * @return the flipped image
     */
    public static NDArray randomFlipTopBottom(NDArray image) {
        return image.getNDArrayInternal().randomFlipTopBottom();
    }

    /**
     * Crop the input image with random scale and aspect ratio.
     *
     * @param image the image with HWC format
     * @param width the output width of the image
     * @param height the output height of the image
     * @param minAreaScale minimum targetArea/srcArea value
     * @param maxAreaScale maximum targetArea/srcArea value
     * @param minAspectRatio minimum aspect ratio
     * @param maxAspectRatio maximum aspect ratio
     * @return the cropped image
     */
    public static NDArray randomResizedCrop(
            NDArray image,
            int width,
            int height,
            double minAreaScale,
            double maxAreaScale,
            double minAspectRatio,
            double maxAspectRatio) {
        Shape shape = image.getShape();
        if (isCHW(image.getShape()) || shape.dimension() == 4) {
            throw new IllegalArgumentException(
                    "randomResizedCrop only support for HWC image format");
        }
        int h = (int) shape.get(0);
        int w = (int) shape.get(1);
        int srcArea = h * w;
        double targetArea =
                minAreaScale * srcArea
                        + (maxAreaScale - minAreaScale) * srcArea * RandomUtils.nextFloat();
        // get ratio from maximum achievable h and w
        double minRatio = (targetArea / h) / h;
        double maxRatio = w / (targetArea / w);
        double[] intersectRatio = {
            Math.max(minRatio, minAspectRatio), Math.min(maxRatio, maxAspectRatio)
        };
        if (intersectRatio[1] < intersectRatio[0]) {
            return centerCrop(image, width, height);
        }
        // compute final area to crop
        float finalRatio =
                RandomUtils.nextFloat((float) intersectRatio[0], (float) intersectRatio[1]);
        int newWidth = (int) Math.round(Math.sqrt(targetArea * finalRatio));
        int newHeight = (int) (newWidth / finalRatio);
        // num in nextInt(num) should be greater than 0
        // otherwise it throws IllegalArgumentException: bound must be positive
        int x = w == newWidth ? 0 : RandomUtils.nextInt(w - newWidth);
        int y = h == newHeight ? 0 : RandomUtils.nextInt(h - newHeight);
        try (NDArray cropped = crop(image, x, y, newWidth, newHeight)) {
            return resize(cropped, width, height);
        }
    }

    /**
     * Randomly jitters image brightness with a factor chosen from [max(0, 1 - brightness), 1 +
     * brightness].
     *
     * @param image the image with HWC format
     * @param brightness the brightness factor from 0 to 1
     * @return the transformed image
     */
    public static NDArray randomBrightness(NDArray image, float brightness) {
        return image.getNDArrayInternal().randomBrightness(brightness);
    }

    /**
     * Randomly jitters image hue with a factor chosen from [max(0, 1 - hue), 1 + hue].
     *
     * @param image the image with HWC format
     * @param hue the hue factor from 0 to 1
     * @return the transformed image
     */
    public static NDArray randomHue(NDArray image, float hue) {
        return image.getNDArrayInternal().randomHue(hue);
    }

    /**
     * Randomly jitters the brightness, contrast, saturation, and hue of an image.
     *
     * @param image the image with HWC format
     * @param brightness the brightness factor from 0 to 1
     * @param contrast the contrast factor from 0 to 1
     * @param saturation the saturation factor from 0 to 1
     * @param hue the hue factor from 0 to 1
     * @return the transformed image
     */
    public static NDArray randomColorJitter(
            NDArray image, float brightness, float contrast, float saturation, float hue) {
        return image.getNDArrayInternal().randomColorJitter(brightness, contrast, saturation, hue);
    }

    /**
     * Check if the shape of the image follows CHW/NCHW.
     *
     * @param shape the shape of the image
     * @return true for (N)CHW, false for (N)HWC
     */
    public static boolean isCHW(Shape shape) {
        if (shape.dimension() < 3) {
            throw new IllegalArgumentException(
                    "Not a valid image shape, require at least three dimensions");
        }
        if (shape.dimension() == 4) {
            shape = shape.slice(1);
        }
        if (shape.get(0) == 1 || shape.get(0) == 3) {
            return true;
        } else if (shape.get(2) == 1 || shape.get(2) == 3) {
            return false;
        }
        throw new IllegalArgumentException("Image is not CHW or HWC");
    }
}
