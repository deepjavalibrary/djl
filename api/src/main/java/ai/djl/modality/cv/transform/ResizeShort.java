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

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Transform;

/** A {@link Transform} that resizes the image and keep aspect ratio. */
public class ResizeShort implements Transform {

    private int shortEdge;
    private int longEdge;
    private Image.Interpolation interpolation;

    /**
     * Creates a {@code ResizeShort} {@link Transform} that resizes to the given size.
     *
     * @param shortEdge the length of the short edge
     */
    public ResizeShort(int shortEdge) {
        this(shortEdge, -1, Image.Interpolation.BILINEAR);
    }

    /**
     * Creates a {@code ResizeShort} {@link Transform} that resizes to the given size and given
     * interpolation.
     *
     * @param shortEdge the length of the short edge
     * @param longEdge the length of the long edge
     * @param interpolation the desired interpolation
     */
    public ResizeShort(int shortEdge, int longEdge, Image.Interpolation interpolation) {
        this.shortEdge = shortEdge;
        this.longEdge = longEdge;
        this.interpolation = interpolation;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transform(NDArray array) {
        Shape shape = array.getShape();
        int width = (int) shape.get(1);
        int height = (int) shape.get(0);
        int min = Math.min(width, height);
        int max = Math.max(width, height);
        int newShort;
        int newLong;

        if (shortEdge < 0) {
            newShort = min * longEdge / max;
            newLong = longEdge;
        } else {
            newShort = shortEdge;
            newLong = max * shortEdge / min;
            if (longEdge > 0 && newLong > longEdge) {
                newShort = min * longEdge / max;
                newLong = longEdge;
            }
        }
        int rescaledWidth;
        int rescaledHeight;
        if (width > height) {
            rescaledWidth = newLong;
            rescaledHeight = newShort;
        } else {
            rescaledWidth = newShort;
            rescaledHeight = newLong;
        }

        return NDImageUtils.resize(array, rescaledWidth, rescaledHeight, interpolation);
    }
}
