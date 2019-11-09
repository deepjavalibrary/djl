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

import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.translate.Transform;

/** A {@link Transform} that crops the image to a given location and size. */
public class Crop implements Transform {

    private int x;
    private int y;
    private int width;
    private int height;

    /**
     * Creates a {@code CenterCrop} {@link Transform}.
     *
     * @param x the x coordinate of the top-left corner of the crop
     * @param y the y coordinate of the top-left corner of the crop
     * @param width the width of the cropped image
     * @param height the height of the cropped image
     */
    public Crop(int x, int y, int width, int height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transform(NDArray array) {
        return NDImageUtils.crop(array, x, y, width, height);
    }
}
