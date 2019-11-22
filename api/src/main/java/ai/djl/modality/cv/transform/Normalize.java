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

/** A {@link Transform} that normalizes an image {@link NDArray} of shape CHW or NCHW. */
public class Normalize implements Transform {
    private float[] mean;
    private float[] std;

    /**
     * Creates a {@code Normalize} {@link Transform} that normalizes.
     *
     * @param mean the mean to normalize with for each channel
     * @param std the standard deviation to normalize with for each channel
     * @see NDImageUtils#normalize(NDArray, float[], float[])
     */
    public Normalize(float[] mean, float[] std) {
        this.mean = mean;
        this.std = std;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transform(NDArray array) {
        return NDImageUtils.normalize(array, mean, std);
    }
}
