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
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.List;
import javax.imageio.ImageIO;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
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

    /**
     * Draws all joints of a body on an image.
     *
     * @param image the input image
     * @param joints the joints of the body
     */
    public static void drawJoints(BufferedImage image, List<Joint> joints) {
        Graphics2D g = (Graphics2D) image.getGraphics();
        int stroke = 2;
        g.setStroke(new BasicStroke(stroke));

        int imageWidth = image.getWidth();
        int imageHeight = image.getHeight();

        for (Joint joint : joints) {
            g.setPaint(randomColor().darker());
            int x = (int) (joint.getX() * imageWidth);
            int y = (int) (joint.getY() * imageHeight);
            g.fillOval(x, y, 10, 10);
        }
        g.dispose();
    }

    private static Color randomColor() {
        return new Color(RandomUtils.nextInt(255));
    }

    /**
     * Converts image to a float buffer.
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
