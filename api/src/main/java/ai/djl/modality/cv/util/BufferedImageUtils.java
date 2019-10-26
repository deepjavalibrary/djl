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
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.RandomUtils;
import java.awt.Color;
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
    private static ByteBuffer toByteBuffer(NDManager manager, BufferedImage image) {
        // Get height and width of the image
        int width = image.getWidth();
        int height = image.getHeight();

        // get an array of integer pixels in the default RGB color mode
        int[] pixels = image.getRGB(0, 0, width, height, null, 0, width);

        // 3 times height and width for R,G,B channels
        ByteBuffer bb = manager.allocateDirect(3 * height * width);
        for (int rgb : pixels) {
            // getting red color
            bb.put((byte) (rgb >> 16));
            // getting green color
            bb.put((byte) (rgb >> 8));
            // getting blue color
            bb.put((byte) rgb);
        }
        bb.rewind();
        return bb;
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
        NDArray rgb = manager.create(new Shape(height, width, 3), DataType.UINT8);
        rgb.set(toByteBuffer(manager, image));
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
        return NDImageUtils.toTensor(toNDArray(manager, image, flag));
    }
}
