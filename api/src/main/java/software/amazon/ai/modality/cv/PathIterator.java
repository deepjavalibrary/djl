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
package software.amazon.ai.modality.cv;

/**
 * The <code>PathIterator</code> interface provides the mechanism for objects that implement the
 * {@link BoundingBox BoundingBox} interface to return the geometry of their boundary. It does this
 * by allowing a caller to retrieve the path of that boundary one segment at a time.
 *
 * @see Rectangle
 * @see BoundingBox
 */
public interface PathIterator {

    /**
     * Tests if the iteration is complete.
     *
     * @return <code>true</code> if all the segments have been read; <code>false</code> otherwise.
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
     * @return the path-segment type of the current path segment.
     */
    Point currentPoint();
}
