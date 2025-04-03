/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.transform;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Transform;

/** A {@link Transform} that pad the image to square. */
public class Pad implements Transform {

    private double value;

    /**
     * Constructs a new {@code Pad} instance.
     *
     * @param value the padding value
     */
    public Pad(double value) {
        this.value = value;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transform(NDArray array) {
        Shape shape = array.getShape();
        int w = (int) shape.get(1);
        int h = (int) shape.get(0);
        if (w == h) {
            return array;
        }
        int max = Math.max(w, h);
        int padW = max - w;
        int padH = max - h;
        Shape padding = new Shape(0, 0, 0, padW, 0, padH);
        array = array.pad(padding, value);
        return array;
    }
}
