/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.input;

/** A possible input for a {@link ai.djl.Application.CV#GAN} model. */
public final class BigGANInput {

    private int categoryId;
    private int sampleSize;
    private float truncation;

    public BigGANInput(int categoryId) {
        this.categoryId = categoryId;
        sampleSize = 1;
        truncation = 0.5f;
    }

    /**
     * Construct a {@code BIgGANInput}, this is needed for preprocessing.
     *
     * @param sampleSize the desired number of image samples
     * @param truncation the level of truncation, level of variability in the output
     */
    public BigGANInput(int categoryId, int sampleSize, float truncation) {
        this(categoryId);
        this.sampleSize = sampleSize;
        this.truncation = truncation;
    }

    /**
     * Returns the categoryId.
     *
     * @return the categoryId
     */
    public int getCategoryId() {
        return categoryId;
    }

    /**
     * Returns the sample size.
     *
     * @return the sample size
     */
    public int getSampleSize() {
        return sampleSize;
    }

    /**
     * Returns the truncation level.
     *
     * @return the truncation level
     */
    public float getTruncation() {
        return truncation;
    }
}
