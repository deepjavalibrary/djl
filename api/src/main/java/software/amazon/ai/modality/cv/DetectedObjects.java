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
package software.amazon.ai.modality.cv;

import java.util.List;
import software.amazon.ai.modality.AbstractClassifications;
import software.amazon.ai.modality.cv.DetectedObjects.Item;

/** A class representing the detected object in an object detection case. */
public class DetectedObjects extends AbstractClassifications<Item> {

    private List<BoundingBox> boundingBoxes;

    public DetectedObjects(
            List<String> classNames, List<Double> probabilities, List<BoundingBox> boundingBoxes) {
        super(classNames, probabilities);
        this.boundingBoxes = boundingBoxes;
    }

    @Override
    protected Item item(int index) {
        return new Item(index);
    }

    public class Item extends AbstractClassifications<Item>.Item {

        protected Item(int index) {
            super(index);
        }

        /**
         * Returns {@link BoundingBox} of detected object.
         *
         * @return {@link BoundingBox} of detected object
         */
        public BoundingBox getBoundingBox() {
            return boundingBoxes.get(index);
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
