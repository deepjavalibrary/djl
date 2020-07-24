/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

/** A {@link Transform} that crop the input image with random scale and aspect ratio. */
public class RandomResizedCrop implements Transform {

    private int width;
    private int height;
    private double minAreaScale;
    private double maxAreaScale;
    private double minAspectRatio;
    private double maxAspectRatio;

    /**
     * Creates a {@code RandomResizedCrop} {@link Transform}.
     *
     * @param width the output width of the image
     * @param height the output height of the image
     * @param minAreaScale minimum targetArea/srcArea value
     * @param maxAreaScale maximum targetArea/srcArea value
     * @param minAspectRatio minimum aspect ratio
     * @param maxAspectRatio maximum aspect ratio
     */
    public RandomResizedCrop(
            int width,
            int height,
            double minAreaScale,
            double maxAreaScale,
            double minAspectRatio,
            double maxAspectRatio) {
        this.width = width;
        this.height = height;
        this.minAreaScale = minAreaScale;
        this.maxAreaScale = maxAreaScale;
        this.minAspectRatio = minAspectRatio;
        this.maxAspectRatio = maxAspectRatio;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transform(NDArray array) {
        return NDImageUtils.randomResizedCrop(
                array, width, height, minAreaScale, maxAreaScale, minAspectRatio, maxAspectRatio);
    }
}
