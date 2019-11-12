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

/**
 * A sequence of points used to outline an object in an image.
 *
 * <p>The primary use is to return the boundaries in a {@link BoundingBox} object.
 *
 * @see Rectangle
 * @see BoundingBox
 */
public interface PathIterator {

    /**
     * Tests if the iteration is complete.
     *
     * @return {@code true} if all the segments have been read; {@code false} otherwise
     */
    boolean hasNext();

    /**
     * Moves the iterator to the next segment of the path forwards along the primary direction of
     * traversal as long as there are more points in that direction.
     */
    void next();

    /**
     * Returns the coordinates and type of the current path segment in the iteration.
     *
     * @return the path-segment type of the current path segment
     */
    Point currentPoint();
}
