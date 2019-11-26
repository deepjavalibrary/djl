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
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import javax.imageio.ImageIO;

/**
 * {@code BufferedImageUtils} is an image processing utility to load, reshape and convert images
 * using {@link BufferedImage}.
 */
public final class BufferedImageUtils {

    static {
        if (System.getProperty("apple.awt.UIElement") == null) {
            // disables coffee cup image showing up on macOS
            System.setProperty("apple.awt.UIElement", "true");
        }
    }

    private BufferedImageUtils() {}

    /**
     * Loads the image from the specified file.
     *
     * @param path the path of the file to be loaded
     * @return a {@link BufferedImage}
     * @throws IOException if file is not found
     */
    public static BufferedImage fromFile(Path path) throws IOException {
        return ImageIO.read(path.toAbsolutePath().toFile());
    }

    /**
     * Loads the image from the specified URL.
     *
     * @param url the URL of the file to be loaded
     * @return a {@link BufferedImage}
     * @throws IOException if url is not found
     */
    public static BufferedImage fromUrl(URL url) throws IOException {
        return ImageIO.read(url);
    }

    /**
     * Loads the image from the specified URL.
     *
     * @param url the URL of the file to be loaded
     * @return a {@link BufferedImage}
     * @throws IOException if url is not found
     */
    public static BufferedImage fromUrl(String url) throws IOException {
        return ImageIO.read(new URL(url));
    }

    /**
     * Returns a random color.
     *
     * @return a random color
     */
    public static Color randomColor() {
        return new Color(RandomUtils.nextInt(255));
    }

    /**
     * Converts a {@code BufferedImage} to an RGB NDArray.
     *
     * @param manager a {@link NDManager} to create the new NDArray with
     * @param image the buffered image to be converted
     * @return the image {@link NDArray}
     */
    public static NDArray toNDArray(NDManager manager, BufferedImage image) {
        return toNDArray(manager, image, null);
    }

    /**
     * Converts a {@code BufferedImage} to an NDArray with designated color mode.
     *
     * @param manager a {@link NDManager} to create the new NDArray with
     * @param image the buffered image to be converted
     * @param flag the color mode
     * @return the image {@link NDArray}
     */
    public static NDArray toNDArray(
            NDManager manager, BufferedImage image, NDImageUtils.Flag flag) {
        int width = image.getWidth();
        int height = image.getHeight();
        int channel;
        if (flag == NDImageUtils.Flag.GRAYSCALE) {
            channel = 1;
        } else {
            channel = 3;
        }

        ByteBuffer bb = manager.allocateDirect(channel * height * width);
        if (image.getType() == BufferedImage.TYPE_BYTE_GRAY) {
            byte[] data = ((DataBufferByte) image.getData().getDataBuffer()).getData();
            for (byte gray : data) {
                bb.put(gray);
                if (flag != NDImageUtils.Flag.GRAYSCALE) {
                    bb.put(gray);
                    bb.put(gray);
                }
            }
        } else {
            // get an array of integer pixels in the default RGB color mode
            int[] pixels = image.getRGB(0, 0, width, height, null, 0, width);
            for (int rgb : pixels) {
                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;

                if (flag == NDImageUtils.Flag.GRAYSCALE) {
                    int gray = (red + green + blue) / 3;
                    bb.put((byte) gray);
                } else {
                    bb.put((byte) red);
                    bb.put((byte) green);
                    bb.put((byte) blue);
                }
            }
        }

        NDArray array = manager.create(new Shape(height, width, channel), DataType.UINT8);
        bb.rewind();
        array.set(bb);
        return array;
    }

    /**
     * Reads an image file and converts to NDArray.
     *
     * <p>Returns an image with Color mode.
     *
     * @param manager a {@link NDManager} to create the new NDArray with
     * @param path the path to the image file
     * @return the image NDArray
     * @throws IOException if file is not found
     */
    public static NDArray readFileToArray(NDManager manager, Path path) throws IOException {
        return readFileToArray(manager, path, null);
    }

    /**
     * Reads an image file and converts to NDArray.
     *
     * @param manager a {@link NDManager} to create the new NDArray with
     * @param path the path to the image file
     * @param flag the color option to load with
     * @return the image NDArray
     * @throws IOException if file is not found
     */
    public static NDArray readFileToArray(NDManager manager, Path path, NDImageUtils.Flag flag)
            throws IOException {
        return toNDArray(manager, fromFile(path), flag);
    }
}
