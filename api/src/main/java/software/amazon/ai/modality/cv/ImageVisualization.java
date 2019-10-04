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
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import software.amazon.ai.modality.cv.Joints.Joint;
import software.amazon.ai.modality.cv.util.BufferedImageUtils;

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

        for (DetectedObjects.Item result : detections.items()) {
            String className = result.getClassName();
            BoundingBox box = result.getBoundingBox();
            g.setPaint(BufferedImageUtils.randomColor().darker());

            box.draw(g, imageWidth, imageHeight);
            Point p = box.getPoint();
            int x = (int) (p.getX() * imageWidth);
            int y = (int) (p.getY() * imageHeight);
            drawText(g, className, x, y, stroke, 4);
        }
        g.dispose();
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
