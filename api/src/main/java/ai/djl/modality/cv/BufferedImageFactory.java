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
package ai.djl.modality.cv;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import javax.imageio.ImageIO;

/** {@code BufferedImageFactory} is the default implementation of {@link ImageFactory}. */
public class BufferedImageFactory implements ImageFactory {

    static {
        if (System.getProperty("apple.awt.UIElement") == null) {
            // disables coffee cup image showing up on macOS
            System.setProperty("apple.awt.UIElement", "true");
        }
    }

    /** {@inheritDoc} */
    @Override
    public Image fromFile(Path path) throws IOException {
        BufferedImage image = ImageIO.read(path.toFile());
        if (image == null) {
            throw new IOException("Failed to read image from: " + path);
        }
        return new BufferedImageWrapper(image);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromUrl(URL url) throws IOException {
        BufferedImage image = ImageIO.read(url);
        if (image == null) {
            throw new IOException("Failed to read image from: " + url);
        }
        return new BufferedImageWrapper(image);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromInputStream(InputStream is) throws IOException {
        BufferedImage image = ImageIO.read(is);
        if (image == null) {
            throw new IOException("Failed to read image from input stream");
        }
        return new BufferedImageWrapper(image);
    }

    /** {@inheritDoc} */
    @Override
    public Image fromImage(Object image) {
        if (!(image instanceof BufferedImage)) {
            throw new IllegalArgumentException("only BufferedImage allowed");
        }
        return new BufferedImageWrapper((BufferedImage) image);
    }

    static class BufferedImageWrapper implements Image {
        private final BufferedImage image;

        BufferedImageWrapper(BufferedImage image) {
            this.image = image;
        }

        /** {@inheritDoc} */
        @Override
        public int getWidth() {
            return image.getWidth();
        }

        /** {@inheritDoc} */
        @Override
        public int getHeight() {
            return image.getHeight();
        }

        /** {@inheritDoc} */
        @Override
        public NDArray toNDArray(NDManager manager, Flag flag) {
            int width = image.getWidth();
            int height = image.getHeight();
            int channel;
            if (flag == Flag.GRAYSCALE) {
                channel = 1;
            } else {
                channel = 3;
            }

            ByteBuffer bb = manager.allocateDirect(channel * height * width);
            if (image.getType() == BufferedImage.TYPE_BYTE_GRAY) {
                byte[] data = ((DataBufferByte) image.getData().getDataBuffer()).getData();
                for (byte gray : data) {
                    bb.put(gray);
                    if (flag != Flag.GRAYSCALE) {
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

                    if (flag == Flag.GRAYSCALE) {
                        int gray = (red + green + blue) / 3;
                        bb.put((byte) gray);
                    } else {
                        bb.put((byte) red);
                        bb.put((byte) green);
                        bb.put((byte) blue);
                    }
                }
            }
            bb.rewind();
            return manager.create(bb, new Shape(height, width, channel), DataType.UINT8);
        }

        /** {@inheritDoc} */
        @Override
        public void save(OutputStream os, String type) throws IOException {
            ImageIO.write(image, type, os);
        }
    }
}
