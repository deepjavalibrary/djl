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

import java.awt.Graphics2D;

/**
 * A {@code Rectangle} specifies an area in a coordinate space that is enclosed by the {@code
 * Rectangle} object's upper-left point {@link Point} in the coordinate space, its width, and its
 * height.
 */
public class Rectangle implements BoundingBox {

    private Point point;
    private double height;
    private double width;

    /**
     * Constructs a new {@code Rectangle} whose upper-left corner is specified as {@code (x,y)} and
     * whose width and height are specified by the arguments of the same name.
     *
     * @param y the specified Y coordinate
     * @param x the specified X coordinate
     * @param height the height of the {@code Rectangle}
     * @param width the width of the {@code Rectangle}
     */
    public Rectangle(double y, double x, double height, double width) {
        this(new Point(x, y), height, width);
    }

    /**
     * Constructs a new {@code Rectangle} whose upper-left corner is specified as coordinate {@code
     * point} and whose width and height are specified by the arguments of the same name.
     *
     * @param point upper-left corner of the coordinate
     * @param height the height of the {@code Rectangle}
     * @param width the width of the {@code Rectangle}
     */
    public Rectangle(Point point, double height, double width) {
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

            @Override
            public boolean hasNext() {
                return index < 4;
            }

            @Override
            public void next() {
                if (index > 3) {
                    throw new IllegalStateException("No more path in iterator.");
                }
                ++index;
            }

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

    @Override
    public Point getPoint() {
        return point;
    }

    @Override
    public void draw(Graphics2D g, int imageHeight, int imageWidth) {
        int x = (int) (getX() * imageWidth);
        int y = (int) (getY() * imageHeight);
        int w = (int) (getWidth() * imageWidth);
        int h = (int) (getHeight() * imageHeight);
        g.drawRect(x, y, w, h);
    }

    public double getX() {
        return point.getX();
    }

    public double getY() {
        return point.getY();
    }

    public double getWidth() {
        return width;
    }

    public double getHeight() {
        return height;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        double x = point.getX();
        double y = point.getY();
        return "[x=" + x + ", y=" + y + ", width=" + width + ", height=" + height + ']';
    }
}
