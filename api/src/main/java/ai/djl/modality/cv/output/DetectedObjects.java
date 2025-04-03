/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.modality.Classifications;

import java.util.List;

/**
 * A class representing the detected objects results for a single image in an {@link
 * ai.djl.Application.CV#OBJECT_DETECTION} case.
 */
public class DetectedObjects extends Classifications {

    private static final long serialVersionUID = 1L;

    @SuppressWarnings("serial")
    private List<BoundingBox> boundingBoxes;

    /**
     * Constructs a DetectedObjects, usually during post-processing.
     *
     * <p>All three inputs(classNames, probabilities, boundingBoxes) should be parallel lists.
     *
     * @param classNames the names of the objects that were detected
     * @param probabilities the probability of the objects that were detected
     * @param boundingBoxes the bounding boxes of the objects that were detected
     */
    public DetectedObjects(
            List<String> classNames, List<Double> probabilities, List<BoundingBox> boundingBoxes) {
        super(classNames, probabilities);
        this.boundingBoxes = boundingBoxes;
        this.topK = Integer.MAX_VALUE;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <T extends Classification> T item(int index) {
        return (T)
                new DetectedObject(
                        classNames.get(index), probabilities.get(index), boundingBoxes.get(index));
    }

    /**
     * Returns the number of objects found in an image.
     *
     * @return the number of objects found in an image
     */
    public int getNumberOfObjects() {
        return boundingBoxes.size();
    }

    /** A {@code DetectedObject} represents a single potential detected Object for an image. */
    public static final class DetectedObject extends Classification {

        private BoundingBox boundingBox;

        /**
         * Constructs a bounding box with the given data.
         *
         * @param className name of the type of object
         * @param probability probability that the object is correct
         * @param boundingBox the location of the object
         */
        public DetectedObject(String className, double probability, BoundingBox boundingBox) {
            super(className, probability);
            this.boundingBox = boundingBox;
        }

        /**
         * Returns the {@link ai.djl.modality.cv.output.BoundingBox} of the detected object.
         *
         * @return the {@link ai.djl.modality.cv.output.BoundingBox} of the detected object
         */
        public BoundingBox getBoundingBox() {
            return boundingBox;
        }

        /** {@inheritDoc} */
        @Override
        public String toString() {
            double probability = getProbability();
            StringBuilder sb = new StringBuilder(200);
            sb.append("{\"className\": \"").append(getClassName()).append("\", \"probability\": ");
            if (probability < 0.00001) {
                sb.append(String.format("%.1e", probability));
            } else {
                probability = (int) (probability * 100000) / 100000f;
                sb.append(String.format("%.5f", probability));
            }
            if (boundingBox != null) {
                sb.append(", \"boundingBox\": ").append(boundingBox);
            }
            sb.append('}');
            return sb.toString();
        }
    }
}
