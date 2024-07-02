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
import java.util.PriorityQueue;

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

    @SuppressWarnings("serial")
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
        Rectangle rect = box.getBounds();

        // computing area of each rectangles
        double s1 = (width + 1) * (height + 1);
        double s2 = (rect.getWidth() + 1) * (rect.getHeight() + 1);
        double sumArea = s1 + s2;

        // find each edge of intersect rectangle
        double left = Math.max(getX(), rect.getX());
        double top = Math.max(getY(), rect.getY());
        double right = Math.min(getX() + getWidth(), rect.getX() + rect.getWidth());
        double bottom = Math.min(getY() + getHeight(), rect.getY() + rect.getHeight());

        // judge if there is a intersect
        if (left > right || top > bottom) {
            return 0.0;
        }

        double intersect = (right - left + 1) * (bottom - top + 1);
        return intersect / (sumArea - intersect);
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
        return String.format(
                "{\"x\"=%.3f, \"y\"=%.3f, \"width\"=%.3f, \"height\"=%.3f}", x, y, width, height);
    }

    /**
     * Applies nms (non-maximum suppression) to the list of rectangles.
     *
     * @param boxes an list of {@code Rectangle}
     * @param scores a list of scores
     * @param nmsThreshold the nms threshold
     * @return the filtered list with the index of the original list
     */
    public static List<Integer> nms(
            List<Rectangle> boxes, List<Double> scores, float nmsThreshold) {
        List<Integer> ret = new ArrayList<>();
        PriorityQueue<Integer> pq =
                new PriorityQueue<>(
                        50,
                        (lhs, rhs) -> {
                            // Intentionally reversed to put high confidence at the head of the
                            // queue.
                            return Double.compare(scores.get(rhs), scores.get(lhs));
                        });
        for (int i = 0; i < boxes.size(); ++i) {
            pq.add(i);
        }

        // do non maximum suppression
        while (!pq.isEmpty()) {
            // insert detection with max confidence
            int[] detections = pq.stream().mapToInt(Integer::intValue).toArray();
            ret.add(detections[0]);
            Rectangle box = boxes.get(detections[0]);
            pq.clear();
            for (int i = 1; i < detections.length; i++) {
                int detection = detections[i];
                Rectangle location = boxes.get(detection);
                if (box.boxIou(location) < nmsThreshold) {
                    pq.add(detection);
                }
            }
        }
        return ret;
    }

    private double boxIou(Rectangle other) {
        double intersection = intersection(other);
        double union =
                getWidth() * getHeight() + other.getWidth() * other.getHeight() - intersection;
        return intersection / union;
    }

    private double intersection(Rectangle b) {
        double w =
                overlap(
                        (getX() * 2 + getWidth()) / 2,
                        getWidth(),
                        (b.getX() * 2 + b.getWidth()) / 2,
                        b.getWidth());
        double h =
                overlap(
                        (getY() * 2 + getHeight()) / 2,
                        getHeight(),
                        (b.getY() * 2 + b.getHeight()) / 2,
                        b.getHeight());
        if (w < 0 || h < 0) {
            return 0;
        }
        return w * h;
    }

    private double overlap(double x1, double w1, double x2, double w2) {
        double l1 = x1 - w1 / 2;
        double l2 = x2 - w2 / 2;
        double left = Math.max(l1, l2);
        double r1 = x1 + w1 / 2;
        double r2 = x2 + w2 / 2;
        double right = Math.min(r1, r2);
        return right - left;
    }
}
