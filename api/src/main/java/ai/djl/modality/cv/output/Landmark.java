/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import java.util.List;

/** {@code Landmark} is the container that stores the key points for landmark on a single face. */
public class Landmark extends Rectangle {

    private static final long serialVersionUID = 1L;

    private List<Point> points;

    /**
     * Constructs a {@code Landmark} using a list of points.
     *
     * @param x the left coordinate of the bounding rectangle
     * @param y the top coordinate of the bounding rectangle
     * @param width the width of the bounding rectangle
     * @param height the height of the bounding rectangle
     * @param points the key points for each face
     */
    public Landmark(double x, double y, double width, double height, List<Point> points) {
        super(x, y, width, height);
        this.points = points;
    }

    /** {@inheritDoc} */
    @Override
    public Iterable<Point> getPath() {
        return points;
    }
}
