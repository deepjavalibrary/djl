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
package ai.djl.modality.cv.transform;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.translate.Transform;

/** A {@link Transform} that resizes the image. */
public class Resize implements Transform {

    private int width;
    private int height;
    private Image.Interpolation interpolation;

    /**
     * Creates a {@code Resize} {@link Transform} that resizes to the given size.
     *
     * @param size the new size to use for both height and width
     */
    public Resize(int size) {
        this(size, size, Image.Interpolation.BILINEAR);
    }

    /**
     * Creates a {@code Resize} {@link Transform} that resizes to the given width and height.
     *
     * @param width the desired width
     * @param height the desired height
     */
    public Resize(int width, int height) {
        this(width, height, Image.Interpolation.BILINEAR);
    }

    /**
     * Creates a {@code Resize} {@link Transform} that resizes to the given width and height with
     * given interpolation.
     *
     * @param width the desired width
     * @param height the desired height
     * @param interpolation the desired interpolation
     */
    public Resize(int width, int height, Image.Interpolation interpolation) {
        this.width = width;
        this.height = height;
        this.interpolation = interpolation;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transform(NDArray array) {
        return NDImageUtils.resize(array, width, height, interpolation);
    }
}
