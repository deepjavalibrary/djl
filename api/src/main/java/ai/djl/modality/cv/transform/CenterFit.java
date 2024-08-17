/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Transform;

/** A {@link Transform} that fit the size of an image. */
public class CenterFit implements Transform {

    private int width;
    private int height;

    /**
     * Creates a {@code CenterFit} {@link Transform} that fit to the given width and height with
     * given interpolation.
     *
     * @param width the desired width
     * @param height the desired height
     */
    public CenterFit(int width, int height) {
        this.width = width;
        this.height = height;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transform(NDArray array) {
        Shape shape = array.getShape();
        int w = (int) shape.get(1);
        int h = (int) shape.get(0);
        if (w > width || h > height) {
            array = NDImageUtils.centerCrop(array, Math.min(w, width), Math.min(h, height));
        }
        int padW = width - w;
        int padH = height - h;
        if (padW > 0 || padH > 0) {
            padW = Math.max(0, padW);
            padH = Math.max(0, padH);
            int padW1 = padW / 2;
            int padH1 = padH / 2;
            Shape padding = new Shape(0, 0, padW1, padW - padW1, padH1, padH - padH1);
            array = array.pad(padding, 0);
        }
        return array;
    }
}
