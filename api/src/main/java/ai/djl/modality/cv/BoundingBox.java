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

/** An interface representing a bounding box for the detected object. */
public interface BoundingBox {

    /**
     * Returns the bounding {@code Rectangle} of this {@code BoundingBox}.
     *
     * @return a new {@code Rectangle} for this {@code BoundingBox}
     */
    Rectangle getBounds();

    /**
     * Returns an iterator object that iterates along the {@code BoundingBox} boundary and provides
     * access to the geometry of the {@code BoundingBox} outline.
     *
     * @return a {@code PathIterator} object, which independently traverses the geometry of the
     *     {@code BoundingBox}
     */
    PathIterator getPath();

    /**
     * Returns the top left point of the bounding box.
     *
     * @return the {@link Point} of the top left corner
     */
    Point getPoint();

    /**
     * Draws the bounding box using the {@link Graphics2D}.
     *
     * @param g the Graphics2D object of the image
     * @param imageWidth the width of the image
     * @param imageHeight the height of the image
     */
    void draw(Graphics2D g, int imageWidth, int imageHeight);
}
