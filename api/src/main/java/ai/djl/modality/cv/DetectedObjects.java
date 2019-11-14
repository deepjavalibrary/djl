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
package ai.djl.modality.cv;

import ai.djl.modality.Classifications;
import java.util.List;

/** A class representing the detected object in an object detection case. */
public class DetectedObjects extends Classifications {

    private List<BoundingBox> boundingBoxes;

    public DetectedObjects(
            List<String> classNames, List<Double> probabilities, List<BoundingBox> boundingBoxes) {
        super(classNames, probabilities);
        this.boundingBoxes = boundingBoxes;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <T extends Classification> T item(int index) {
        return (T)
                new DetectedObject(
                        classNames.get(index), probabilities.get(index), boundingBoxes.get(index));
    }

    public int getNumberOfObjects() {
        return boundingBoxes.size();
    }

    public static final class DetectedObject extends Classification {

        private BoundingBox boundingBox;

        public DetectedObject(String className, double probability, BoundingBox boundingBox) {
            super(className, probability);
            this.boundingBox = boundingBox;
        }

        /**
         * Returns the {@link BoundingBox} of the detected object.
         *
         * @return the {@link BoundingBox} of the detected object
         */
        public BoundingBox getBoundingBox() {
            return boundingBox;
        }

        /** {@inheritDoc} */
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(super.toString());
            if (getBoundingBox() != null) {
                sb.append(", bounds: ").append(getBoundingBox());
            }
            return sb.toString();
        }
    }
}
