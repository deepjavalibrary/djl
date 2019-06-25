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
package com.amazon.ai.inference;

import com.amazon.ai.image.BoundingBox;

/** A class represent detected object in object detection case. */
public class DetectedObject extends Classification {

    private BoundingBox boundingBox;

    /**
     * Construct a <code>DetectedObject</code> instance with <code>className</code>, <code>
     * probability</code> and <code>boundingBox</code>.
     *
     * @param className the class name
     * @param probability the probability
     * @param boundingBox {@link BoundingBox} of detected object
     */
    public DetectedObject(String className, double probability, BoundingBox boundingBox) {
        super(className, probability);
        this.boundingBox = boundingBox;
    }

    /**
     * Retruns {@link BoundingBox} of detected object.
     *
     * @return {@link BoundingBox} of detected object
     */
    public BoundingBox getBoundingBox() {
        return boundingBox;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString());
        if (boundingBox != null) {
            sb.append(", bounds: ").append(boundingBox);
        }
        return sb.toString();
    }
}
