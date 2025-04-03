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

import ai.djl.util.JsonUtils;

import java.io.Serializable;

/**
 * A point representing a location in {@code (x,y)} coordinate space, specified in double precision.
 */
public class Point implements Serializable {

    private static final long serialVersionUID = 1L;

    private double x;
    private double y;

    /**
     * Constructs and initializes a point at the specified {@code (x,y)} location in the coordinate
     * space.
     *
     * @param x the X coordinate of the newly constructed {@code Point}
     * @param y the Y coordinate of the newly constructed {@code Point}
     */
    public Point(double x, double y) {
        this.x = x;
        this.y = y;
    }

    /**
     * Returns the X coordinate of this {@code Point} in {@code double} precision.
     *
     * @return the X coordinate of this {@code Point}
     */
    public double getX() {
        return x;
    }

    /**
     * Returns the Y coordinate of this {@code Point} in {@code double} precision.
     *
     * @return the Y coordinate of this {@code Point}
     */
    public double getY() {
        return y;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return JsonUtils.GSON_COMPACT.toJson(this);
    }
}
