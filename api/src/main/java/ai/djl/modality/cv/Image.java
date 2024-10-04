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

import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Point;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

/**
 * {@code Image} is a container of an image in DJL. The storage type of the image depends on the
 * platform you are running on.
 */
public interface Image {

    /**
     * Gets the width of the image.
     *
     * @return pixels representing width
     */
    int getWidth();

    /**
     * Gets the height of the image.
     *
     * @return pixels representing height
     */
    int getHeight();

    /**
     * Gets the wrapped image.
     *
     * @return the wrapped image object
     */
    Object getWrappedImage();

    /**
     * Creates a new resized image.
     *
     * @param width the new image's desired width
     * @param height the new image's desired height
     * @param copy false to return original image if size is the same
     * @return the new resized image.
     */
    Image resize(int width, int height, boolean copy);

    /**
     * Returns a new {@code Image} of masked area.
     *
     * @param mask the mask for each pixel in the image
     * @return the mask image.
     */
    Image getMask(int[][] mask);

    /**
     * Gets the subimage defined by a specified rectangular region.
     *
     * @param x the X coordinate of the upper-left corner of the specified rectangular region
     * @param y the Y coordinate of the upper-left corner of the specified rectangular region
     * @param w the width of the specified rectangular region
     * @param h the height of the specified rectangular region
     * @return subimage of this image
     */
    Image getSubImage(int x, int y, int w, int h);

    /**
     * Gets a deep copy of the original image.
     *
     * @return the copy of the original image.
     */
    Image duplicate();

    /**
     * Converts image to a RGB {@link NDArray}.
     *
     * @param manager a {@link NDManager} to create the new NDArray with
     * @return {@link NDArray}
     */
    default NDArray toNDArray(NDManager manager) {
        return toNDArray(manager, null);
    }

    /**
     * Converts image to a {@link NDArray}.
     *
     * @param manager a {@link NDManager} to create the new NDArray with
     * @param flag the color mode
     * @return {@link NDArray}
     */
    NDArray toNDArray(NDManager manager, Flag flag);

    /**
     * Save the image to file.
     *
     * @param os {@link OutputStream} to save the image.
     * @param type type of the image, such as "png", "jpeg"
     * @throws IOException image cannot be saved through output stream
     */
    void save(OutputStream os, String type) throws IOException;

    /**
     * Find bounding boxes from a masked image with 0/1 or 0/255.
     *
     * @return the List of bounding boxes of the images
     */
    List<BoundingBox> findBoundingBoxes();

    /**
     * Draws the bounding boxes on the image.
     *
     * @param detections the object detection results
     */
    default void drawBoundingBoxes(DetectedObjects detections) {
        drawBoundingBoxes(detections, -1);
    }

    /**
     * Draws the bounding boxes on the image.
     *
     * @param detections the object detection results
     */
    void drawBoundingBoxes(DetectedObjects detections, float opacity);

    /**
     * Draws a rectangle on the image.
     *
     * @param rect the rectangle to draw
     * @param rgb the color
     * @param thickness the thickness
     */
    void drawRectangle(Rectangle rect, int rgb, int thickness);

    /**
     * Draws a mark on the image.
     *
     * @param points as list of {@code Point}
     */
    default void drawMarks(List<Point> points) {
        int w = getWidth();
        int h = getHeight();
        int size = Math.min(w, h) / 50;
        drawMarks(points, size);
    }

    /**
     * Draws a mark on the image.
     *
     * @param points as list of {@code Point}
     * @param size the radius of the star mark
     */
    void drawMarks(List<Point> points, int size);

    /**
     * Draws all joints of a body on an image.
     *
     * @param joints the joints of the body
     */
    void drawJoints(Joints joints);

    /**
     * Draws the overlay on the image.
     *
     * @param overlay the overlay image
     * @param resize true to resize the overlay image to match the image
     */
    void drawImage(Image overlay, boolean resize);

    /**
     * Creates a star shape.
     *
     * @param point the coordinate
     * @param radius the radius
     * @return the polygon points
     */
    default int[][] createStar(Point point, int radius) {
        int[][] ret = new int[2][10];
        double midX = point.getX();
        double midY = point.getY();
        double[] ratio = {radius, radius * 0.38196601125};

        double delta = Math.PI / 5;
        for (int i = 0; i < 10; ++i) {
            double angle = i * delta;
            double r = ratio[i % 2];
            double x = Math.cos(angle) * r;
            double y = Math.sin(angle) * r;

            ret[0][i] = (int) (x + midX);
            ret[1][i] = (int) (y + midY);
        }
        return ret;
    }

    /** Flag indicates the color channel options for images. */
    enum Flag {
        GRAYSCALE,
        COLOR;

        /**
         * Returns the number of channels for this flag.
         *
         * @return the number of channels for this flag
         */
        public int numChannels() {
            switch (this) {
                case GRAYSCALE:
                    return 1;
                case COLOR:
                    return 3;
                default:
                    throw new IllegalArgumentException("Invalid FLAG");
            }
        }
    }

    /** Interpolation indicates the Interpolation options for resizinig an image. */
    enum Interpolation {
        NEAREST,
        BILINEAR,
        AREA,
        BICUBIC
    }
}
