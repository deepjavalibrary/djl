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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.RandomUtils;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import javax.imageio.ImageIO;

/**
 * {@code BufferedImageUtils} is an image processing utility that can load, reshape and convert
 * images using {@link BufferedImage}.
 */
public final class BufferedImageUtils {

    private static final NDImageUtils.Flag DEFAULT_FLAG = NDImageUtils.Flag.COLOR;

    static {
        if (System.getProperty("apple.awt.UIElement") == null) {
            // disable annoying coffee cup show up on macos
            System.setProperty("apple.awt.UIElement", "true");
        }
    }

    private BufferedImageUtils() {}

    /**
     * Loads the image from the file specified.
     *
     * @param path the file path to be loaded
     * @return a {@link BufferedImage}
     * @throws IOException file is not found
     */
    public static BufferedImage fromFile(Path path) throws IOException {
        return ImageIO.read(path.toFile());
    }

    /**
     * Resizes the image with new width and height.
     *
     * @param image the input image
     * @param newWidth the new width of the reshaped image
     * @param newHeight the new height of the reshaped image
     * @return reshaped {@link BufferedImage}
     */
    public static BufferedImage resize(BufferedImage image, int newWidth, int newHeight) {
        BufferedImage resizedImage =
                new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(image, 0, 0, newWidth, newHeight, null);
        g.dispose();

        return resizedImage;
    }

    public static BufferedImage centerCrop(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();

        if (w == h) {
            return img;
        }

        if (w > h) {
            return centerCrop(img, h, h);
        }

        return centerCrop(img, w, w);
    }

    public static BufferedImage centerCrop(BufferedImage img, int width, int height) {
        BufferedImage ret = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        int w = img.getWidth();
        int h = img.getHeight();

        int sx1;
        int sx2;
        int sy1;
        int sy2;
        int dx1;
        int dx2;
        int dy1;
        int dy2;
        int dw = (w - width) / 2;
        if (dw > 0) {
            sx1 = dw;
            sx2 = sx1 + width;
            dx1 = 0;
            dx2 = width;
        } else {
            sx1 = 0;
            sx2 = w;
            dx1 = -dw;
            dx2 = dx1 + w;
        }
        int dh = (h - height) / 2;
        if (dh > 0) {
            sy1 = dh;
            sy2 = sy1 + height;
            dy1 = 0;
            dy2 = height;
        } else {
            sy1 = 0;
            sy2 = h;
            dy1 = -dh;
            dy2 = dy1 + w;
        }

        Graphics2D g = ret.createGraphics();
        g.drawImage(img, dx1, dy1, dx2, dy2, sx1, sy1, sx2, sy2, null);
        g.dispose();

        return ret;
    }

    public static Color randomColor() {
        return new Color(RandomUtils.nextInt(255));
    }

    /**
     * Converts image to an RGB float buffer.
     *
     * @param manager {@link NDManager} to allocate direct buffer
     * @param image the buffered image to be converted
     * @return {@link FloatBuffer}
     */
    public static FloatBuffer toFloatBuffer(NDManager manager, BufferedImage image) {
        // Get height and width of the image
        int width = image.getWidth();
        int height = image.getHeight();

        // get an array of integer pixels in the default RGB color mode
        int[] pixels = image.getRGB(0, 0, width, height, null, 0, width);

        // 3 times height and width for R,G,B channels
        ByteBuffer bb = manager.allocateDirect(4 * 3 * height * width);
        FloatBuffer buf = bb.asFloatBuffer();
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                int rgb = pixels[row * width + col];

                // getting red color
                buf.put(row * width + col, (rgb >> 16) & 0xFF);
                // getting green color
                buf.put(height * width + row * width + col, (rgb >> 8) & 0xFF);
                // getting blue color
                buf.put(2 * height * width + row * width + col, rgb & 0xFF);
            }
        }
        return buf;
    }

    /**
     * Converts {@code BufferedImage} to RGB NDArray.
     *
     * @param manager {@link NDManager} to create new NDArray with
     * @param image the buffered image to be converted
     * @return {@link NDArray}
     */
    public static NDArray toNDArray(NDManager manager, BufferedImage image) {
        return toNDArray(manager, image, DEFAULT_FLAG);
    }

    /**
     * Converts {@code BufferedImage} to NDArray with designated color mode.
     *
     * @param manager {@link NDManager} to create new NDArray with
     * @param image the buffered image to be converted
     * @param flag The color mode
     * @return {@link NDArray}
     */
    public static NDArray toNDArray(
            NDManager manager, BufferedImage image, NDImageUtils.Flag flag) {
        int width = image.getWidth();
        int height = image.getHeight();
        NDArray rgb = manager.create(toFloatBuffer(manager, image), new Shape(3, height, width));
        if (flag == NDImageUtils.Flag.COLOR) {
            return rgb;
        } else if (flag == NDImageUtils.Flag.GRAYSCALE) {
            return rgb.mean(new int[] {0});
        } else {
            throw new IllegalArgumentException("Cannot convert to NDArray with flag: " + flag);
        }
    }

    public static NDArray readFileToArray(NDManager manager, Path path) throws IOException {
        return readFileToArray(manager, path, DEFAULT_FLAG);
    }

    public static NDArray readFileToArray(NDManager manager, Path path, NDImageUtils.Flag flag)
            throws IOException {
        BufferedImage image = fromFile(path);
        return toNDArray(manager, image, flag);
    }
}
