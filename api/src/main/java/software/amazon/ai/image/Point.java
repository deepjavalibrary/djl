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
package software.amazon.ai.image;

/**
 * A point representing a location in {@code (x,y)} coordinate space, specified in double precision.
 */
public class Point {

    private int x;
    private int y;

    /**
     * Constructs and initializes a point at the specified {@code (x,y)} location in the coordinate
     * space.
     *
     * @param x the X coordinate of the newly constructed <code>Point</code>
     * @param y the Y coordinate of the newly constructed <code>Point</code>
     */
    public Point(int x, int y) {
        this.x = x;
        this.y = y;
    }

    /**
     * Returns the X coordinate of this <code>Point</code> in <code>double</code> precision.
     *
     * @return the X coordinate of this <code>Point</code>.
     */
    public int getX() {
        return x;
    }

    /**
     * Returns the Y coordinate of this <code>Point</code> in <code>double</code> precision.
     *
     * @return the Y coordinate of this <code>Point</code>.
     */
    public int getY() {
        return y;
    }
}
