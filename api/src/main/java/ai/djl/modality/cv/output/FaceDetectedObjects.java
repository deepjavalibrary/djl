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

import java.util.List;

/**
 * A class representing the detected face objects results for a single image in an {@link
 * ai.djl.Application.CV#OBJECT_DETECTION} case.
 */
public class FaceDetectedObjects extends DetectedObjects {
    private static final long serialVersionUID = 1L;
    private List<Landmark> landmarks;

    /**
     * Constructs a FaceDetectedObjects, usually during post-processing.
     *
     * <p>All four inputs(classNames, probabilities, boundingBoxes, landmarks) should be parallel
     * lists.
     *
     * @param classNames the names of the face objects that were detected
     * @param probabilities the probability of the face objects that were detected
     * @param boundingBoxes the bounding boxes of the face objects that were detected
     * @param landmarks the landmarks of the face objects that were detected
     */
    public FaceDetectedObjects(
            List<String> classNames,
            List<Double> probabilities,
            List<BoundingBox> boundingBoxes,
            List<Landmark> landmarks) {
        super(classNames, probabilities, boundingBoxes);
        this.landmarks = landmarks;
    }

    /**
     * Returns the landmarks of face objects found in an image.
     *
     * @return the landmarks of face objects found in an image
     */
    public List<Landmark> getLandmarks() {
        return landmarks;
    }
}
