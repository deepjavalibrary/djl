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
package software.amazon.ai.modality.cv;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.List;
import javax.imageio.ImageIO;
import software.amazon.ai.util.RandomUtils;

/** {@code Images} is an image processing utility that can load, reshape and convert images. */
public final class Images {

    static {
        if (System.getProperty("apple.awt.UIElement") == null) {
            // disable annoying coffee cup show up on macos
            System.setProperty("apple.awt.UIElement", "true");
        }
    }

    private Images() {}

    /**
     * Loads the image from the file specified.
     *
     * @param file the file to be loaded
     * @return a {@link BufferedImage}
     * @throws IOException file is not found
     */
    public static BufferedImage loadImageFromFile(Path file) throws IOException {
        return ImageIO.read(file.toFile());
    }

    /**
     * Resizes the image with new width and height.
     *
     * @param image the input image
     * @param newWidth the new width of the reshaped image
     * @param newHeight the new height of the reshaped image
     * @return reshaped {@link BufferedImage}
     */
    public static BufferedImage resizeImage(BufferedImage image, int newWidth, int newHeight) {
        BufferedImage resizedImage =
                new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(image, 0, 0, newWidth, newHeight, null);
        g.dispose();

        return resizedImage;
    }

    /**
     * Draws the bounding box on an image.
     *
     * @param image the input image
     * @param detections the object detection results
     */
    public static void drawBoundingBox(BufferedImage image, List<DetectedObject> detections) {
        Graphics2D g = (Graphics2D) image.getGraphics();
        int stroke = 2;
        g.setStroke(new BasicStroke(stroke));

        int imageWidth = image.getWidth();
        int imageHeight = image.getHeight();

        for (DetectedObject result : detections) {
            String className = result.getClassName();
            Rectangle rect = result.getBoundingBox().getBounds();
            g.setPaint(randomColor().darker());

            int x = (int) (rect.getX() * imageWidth);
            int y = (int) (rect.getY() * imageHeight);
            int w = (int) (rect.getWidth() * imageWidth);
            int h = (int) (rect.getHeight() * imageHeight);

            g.drawRect(x, y, w, h);
            drawText(g, className, x, y, stroke, 4);
        }
        g.dispose();
    }

    private static Color randomColor() {
        return new Color(RandomUtils.nextInt(255));
    }

    /**
     * Converts image to a float buffer.
     *
     * @param image the buffered image to be converted
     * @return {@link FloatBuffer}
     */
    public static FloatBuffer toFloatBuffer(BufferedImage image) {
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

    private static void drawText(Graphics2D g, String text, int x, int y, int stroke, int padding) {
        FontMetrics metrics = g.getFontMetrics();
        x += stroke / 2;
        y += stroke / 2;
        int width = metrics.stringWidth(text) + padding * 2 - stroke / 2;
        int height = metrics.getHeight() + metrics.getDescent();
        int ascent = metrics.getAscent();
        java.awt.Rectangle background = new java.awt.Rectangle(x, y, width, height);
        g.fill(background);
        g.setPaint(Color.WHITE);
        g.drawString(text, x + padding, y + ascent);
    }
}
