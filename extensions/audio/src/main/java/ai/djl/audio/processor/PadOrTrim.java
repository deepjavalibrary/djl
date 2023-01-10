/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.audio.processor;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/** Pad or trim the samples to the desired numbers. */
public class PadOrTrim implements AudioProcessor {

    private int desiredSamples;

    /**
     * Pad or trim the samples to the fixed length.
     *
     * @param desiredSamples the desired sample points
     */
    public PadOrTrim(int desiredSamples) {
        this.desiredSamples = desiredSamples;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray extractFeatures(NDManager manager, NDArray samples) {
        if (samples.getShape().dimension() != 1) {
            throw new UnsupportedOperationException("Batch samples not supported.");
        }
        long sampleLength = samples.getShape().get(0);
        if (sampleLength > desiredSamples) {
            return samples.get(":" + desiredSamples);
        }
        return samples.concat(manager.zeros(new Shape(desiredSamples - sampleLength)));
    }
}
