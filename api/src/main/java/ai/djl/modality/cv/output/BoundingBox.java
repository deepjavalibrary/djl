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

import java.io.Serializable;

/** An interface representing a bounding box around an object inside an image. */
public interface BoundingBox extends Serializable {

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
     * @return a {@code Iterable} object, which independently traverses the geometry of the {@code
     *     BoundingBox}
     */
    Iterable<Point> getPath();

    /**
     * Returns the top left point of the bounding box.
     *
     * @return the {@link Point} of the top left corner
     */
    Point getPoint();

    /**
     * Gets the Intersection over Union (IoU) value between bounding boxes.
     *
     * <p>Also known as <a href="https://en.wikipedia.org/wiki/Jaccard_index">Jaccard index</a>
     *
     * @param box the bounding box to calculate
     * @return the IoU value
     */
    double getIoU(BoundingBox box);
}
