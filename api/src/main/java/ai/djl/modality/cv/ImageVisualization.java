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
package ai.djl.modality.cv;

import ai.djl.modality.cv.Joints.Joint;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.util.RandomUtils;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.List;

/** A Collection of utilities for visualizing the results of Computer Vision tasks. */
public final class ImageVisualization {

    private ImageVisualization() {}

    /**
     * Draws the bounding boxes on an image.
     *
     * @param image the input image
     * @param detections the object detection results
     */
    public static void drawBoundingBoxes(BufferedImage image, DetectedObjects detections) {
        Graphics2D g = (Graphics2D) image.getGraphics();
        int stroke = 2;
        g.setStroke(new BasicStroke(stroke));
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int imageWidth = image.getWidth();
        int imageHeight = image.getHeight();

        List<DetectedObjects.DetectedObject> list = detections.items();
        for (DetectedObjects.DetectedObject result : list) {
            String className = result.getClassName();
            BoundingBox box = result.getBoundingBox();
            g.setPaint(BufferedImageUtils.randomColor().darker());

            box.draw(g, imageWidth, imageHeight);
            Point p = box.getPoint();
            int x = (int) (p.getX() * imageWidth);
            int y = (int) (p.getY() * imageHeight);
            drawText(g, className, x, y, stroke, 4);
            // If we have a mask instead of a plain rectangle, draw tha mask
            if (box instanceof Mask) {
                Mask mask = (Mask) box;
                drawMask(image, mask);
            }
        }
        g.dispose();
    }

    /**
     * Draws alpha masks on segmented items in image.
     *
     * @param image Buffered image to draw masks upon.
     * @param mask Mask using which the parameters are added
     */
    private static void drawMask(BufferedImage image, Mask mask) {
        float r = RandomUtils.nextFloat();
        float g = RandomUtils.nextFloat();
        float b = RandomUtils.nextFloat();
        int imageWidth = image.getWidth();
        int imageHeight = image.getHeight();
        int x = (int) (mask.getX() * imageWidth);
        int y = (int) (mask.getY() * imageHeight);
        float[][] probDist = mask.getProbDist();
        // Correct some coordinates of box when going out of image
        if (x < 0) {
            x = 0;
        }
        if (y < 0) {
            y = 0;
        }

        BufferedImage maskImage =
                new BufferedImage(probDist.length, probDist[0].length, BufferedImage.TYPE_INT_ARGB);
        for (int xCor = 0; xCor < probDist.length; xCor++) {
            for (int yCor = 0; yCor < probDist[xCor].length; yCor++) {
                float opacity = probDist[xCor][yCor];
                if (opacity < 0.1) {
                    opacity = 0f;
                }
                if (opacity > 0.8) {
                    opacity = 0.8f;
                }
                maskImage.setRGB(xCor, yCor, new Color(r, g, b, opacity).darker().getRGB());
            }
        }
        Graphics2D gR = (Graphics2D) image.getGraphics();
        gR.drawImage(maskImage, x, y, null);
        gR.dispose();
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

    /**
     * Draws all joints of a body on an image.
     *
     * @param image the input image
     * @param joints the joints of the body
     */
    public static void drawJoints(BufferedImage image, Joints joints) {
        Graphics2D g = (Graphics2D) image.getGraphics();
        int stroke = 2;
        g.setStroke(new BasicStroke(stroke));

        int imageWidth = image.getWidth();
        int imageHeight = image.getHeight();

        for (Joint joint : joints.getJoints()) {
            g.setPaint(BufferedImageUtils.randomColor().darker());
            int x = (int) (joint.getX() * imageWidth);
            int y = (int) (joint.getY() * imageHeight);
            g.fillOval(x, y, 10, 10);
        }
        g.dispose();
    }
}
