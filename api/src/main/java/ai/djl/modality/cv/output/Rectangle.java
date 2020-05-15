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

/**
 * A {@code Rectangle} specifies an area in a coordinate space that is enclosed by the {@code
 * Rectangle} object's upper-left point {@link Point} in the coordinate space, its width, and its
 * height.
 */
public class Rectangle implements BoundingBox {

    private static final long serialVersionUID = 1L;
    private Point point;
    private double width;
    private double height;

    /**
     * Constructs a new {@code Rectangle} whose upper-left corner is specified as {@code (x,y)} and
     * whose width and height are specified by the arguments of the same name.
     *
     * @param x the specified X coordinate
     * @param y the specified Y coordinate
     * @param width the width of the {@code Rectangle}
     * @param height the height of the {@code Rectangle}
     */
    public Rectangle(double x, double y, double width, double height) {
        this(new Point(x, y), width, height);
    }

    /**
     * Constructs a new {@code Rectangle} whose upper-left corner is specified as coordinate {@code
     * point} and whose width and height are specified by the arguments of the same name.
     *
     * @param point the upper-left corner of the coordinate
     * @param width the width of the {@code Rectangle}
     * @param height the height of the {@code Rectangle}
     */
    public Rectangle(Point point, double width, double height) {
        this.point = point;
        this.width = width;
        this.height = height;
    }

    /** {@inheritDoc} */
    @Override
    public Rectangle getBounds() {
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public PathIterator getPath() {
        return new PathIterator() {

            private int index;

            /** {@inheritDoc} */
            @Override
            public boolean hasNext() {
                return index < 4;
            }

            /** {@inheritDoc} */
            @Override
            public void next() {
                if (index > 3) {
                    throw new IllegalStateException("No more path in iterator.");
                }
                ++index;
            }

            /** {@inheritDoc} */
            @Override
            public Point currentPoint() {
                switch (index) {
                    case 0:
                        return point;
                    case 1:
                        return new Point(point.getX() + width, point.getY());
                    case 2:
                        return new Point(point.getX() + width, point.getY() + height);
                    case 3:
                        return new Point(point.getX(), point.getY() + height);
                    default:
                        throw new AssertionError("Invalid index: " + index);
                }
            }
        };
    }

    /** {@inheritDoc} */
    @Override
    public Point getPoint() {
        return point;
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
     * @return the left x-coordinate of the Rectangle
     */
    public double getX() {
        return point.getX();
    }

    /**
     * Returns the top y-coordinate of the Rectangle.
     *
     * @return the top y-coordinate of the Rectangle
     */
    public double getY() {
        return point.getY();
    }

    /**
     * Returns the width of the Rectangle.
     *
     * @return the width of the Rectangle
     */
    public double getWidth() {
        return width;
    }

    /**
     * Returns the height of the Rectangle.
     *
     * @return the height of the Rectangle
     */
    public double getHeight() {
        return height;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        double x = point.getX();
        double y = point.getY();
        return String.format("[x=%.3f, y=%.3f, width=%.3f, height=%.3f]", x, y, width, height);
    }
}
