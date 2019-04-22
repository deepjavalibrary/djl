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
package org.apache.mxnet.image;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;

public final class Image {

    private Image() {}

    public static BufferedImage loadImageFromFile(String path) throws IOException {
        return ImageIO.read(new File(path));
    }

    public static BufferedImage reshapeImage(BufferedImage image, int newWidth, int newHeight) {
        BufferedImage resizedImage =
                new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(image, 0, 0, newWidth, newHeight, null);
        g.dispose();

        return resizedImage;
    }

    public static List<BufferedImage> loadInputBatch(List<String> images) throws IOException {
        List<BufferedImage> list = new ArrayList<>(images.size());
        for (String path : images) {
            list.add(loadImageFromFile(path));
        }
        return list;
    }

    public static FloatBuffer toDirectBuffer(BufferedImage image) {
        // Get height and width of the image
        int width = image.getWidth();
        int height = image.getHeight();

        // get an array of integer pixels in the default RGB color mode
        int[] pixels = image.getRGB(0, 0, width, height, null, 0, width);

        // 3 times height and width for R,G,B channels
        ByteBuffer bb = ByteBuffer.allocateDirect(4 * 3 * height * width);
        bb.order(ByteOrder.LITTLE_ENDIAN);
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
}
