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
package ai.djl.modality.cv.output;

import java.util.ArrayList;
import java.util.List;

/**
 * A {@code Rectangle} specifies an area in a coordinate space that is enclosed by the {@code
 * Rectangle} object's upper-left point {@link Point} in the coordinate space, its width, and its
 * height.
 *
 * <p>The rectangle coordinates are usually from 0-1 and are ratios of the image size. For example,
 * if you have an image width of 400 pixels and the rectangle starts at 100 pixels, you would use
 * .25.
 */
public class Rectangle implements BoundingBox {

    private static final long serialVersionUID = 1L;

    private List<Point> corners;

    private double width;
    private double height;

    /**
     * Constructs a new {@code Rectangle} whose upper-left corner is specified as {@code (x,y)} and
     * whose width and height are specified by the arguments of the same name.
     *
     * @param x the specified X coordinate (0-1)
     * @param y the specified Y coordinate (0-1)
     * @param width the width of the {@code Rectangle} (0-1)
     * @param height the height of the {@code Rectangle} (0-1)
     */
    public Rectangle(double x, double y, double width, double height) {
        this(new Point(x, y), width, height);
    }

    /**
     * Constructs a new {@code Rectangle} whose upper-left corner is specified as coordinate {@code
     * point} and whose width and height are specified by the arguments of the same name.
     *
     * @param point the upper-left corner of the coordinate (0-1)
     * @param width the width of the {@code Rectangle} (0-1)
     * @param height the height of the {@code Rectangle} (0-1)
     */
    public Rectangle(Point point, double width, double height) {
        this.width = width;
        this.height = height;
        corners = new ArrayList<>(4);
        corners.add(point);
        corners.add(new Point(point.getX() + width, point.getY()));
        corners.add(new Point(point.getX() + width, point.getY() + height));
        corners.add(new Point(point.getX(), point.getY() + height));
    }

    /** {@inheritDoc} */
    @Override
    public Rectangle getBounds() {
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Iterable<Point> getPath() {
        return corners;
    }

    /** {@inheritDoc} */
    @Override
    public Point getPoint() {
        return corners.get(0);
    }

    /** {@inheritDoc} */
    @Override
    public double getIoU(BoundingBox box) {
        Rectangle rec = (Rectangle) box;
        // caculate intesection lrtb
        double left = Math.max(getX(), rec.getX());
        double top = Math.min(getY(), rec.getY());
        double right = Math.min(getX() + getWidth(), rec.getX() + rec.getWidth());
        double bottom = Math.min(getY() + getHeight(), rec.getY() + rec.getHeight());
        double intersection = (right - left) * (bottom - top);
        return intersection
                / (getWidth() * getHeight() + rec.getWidth() * rec.getHeight() - intersection);
    }

    /**
     * Returns the left x-coordinate of the Rectangle.
     *
     * @return the left x-coordinate of the Rectangle (0-1)
     */
    public double getX() {
        return getPoint().getX();
    }

    /**
     * Returns the top y-coordinate of the Rectangle.
     *
     * @return the top y-coordinate of the Rectangle (0-1)
     */
    public double getY() {
        return getPoint().getY();
    }

    /**
     * Returns the width of the Rectangle.
     *
     * @return the width of the Rectangle (0-1)
     */
    public double getWidth() {
        return width;
    }

    /**
     * Returns the height of the Rectangle.
     *
     * @return the height of the Rectangle (0-1)
     */
    public double getHeight() {
        return height;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        double x = getX();
        double y = getY();
        return String.format("[x=%.3f, y=%.3f, width=%.3f, height=%.3f]", x, y, width, height);
    }
}
