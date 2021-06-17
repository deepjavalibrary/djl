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
    private int sampleSize;
    private float truncation;
    private ImageNetCategory category;

    /**
     * Construct a {@code BIgGANInput}, this is needed for preprocessing.
     *
     * @param sampleSize the desired number of image samples
     * @param truncation the level of truncation, level of variability in the output
     * @param category the desired ImageNet class
     */
    public BigGANInput(int sampleSize, float truncation, ImageNetCategory category) {
        this.sampleSize = sampleSize;
        this.truncation = truncation;
        this.category = category;
    }

    /**
     * Constructs a {@code BIgGANInput} from a builder.
     *
     * @param builder the data to build with
     */
    BigGANInput(Builder builder) {
        this.sampleSize = builder.sampleSize;
        this.truncation = builder.truncation;
        this.category = builder.category;
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

    /**
     * Returns the desired ImageNet class.
     *
     * @return the desired ImageNet class
     */
    public ImageNetCategory getCategory() {
        return category;
    }

    /**
     * Create a builder to build a {@code BIgGANInput} object.
     *
     * @return a builder to build a {@code BIgGANInput} object
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A Builder to construct a {@code BIgGANInput}. */
    public static final class Builder {
        private int sampleSize = 5;
        private float truncation = 0.5f;
        private ImageNetCategory category;

        Builder() {}

        /**
         * Set the optional property sample size.
         *
         * @param sampleSize the desired number of image samples
         * @return the builder to build a {@code BIgGANInput} object
         */
        public Builder optSampleSize(int sampleSize) {
            this.sampleSize = sampleSize;
            return this;
        }

        /**
         * Set the optional property truncation.
         *
         * @param truncation the level of truncation, level of variability in the output
         * @return the builder to build a {@code BIgGANInput} object
         */
        public Builder optTruncation(float truncation) {
            this.truncation = truncation;
            return this;
        }

        /**
         * Set the property category.
         *
         * @param category the desired ImageNet class
         * @return the builder to build a {@code BIgGANInput} object
         */
        public Builder setCategory(ImageNetCategory category) {
            this.category = category;
            return this;
        }

        /**
         * Creates the {@code BIgGANInput} object.
         *
         * @return the {@code BIgGANInput} object
         */
        public BigGANInput build() {
            return new BigGANInput(this);
        }
    }
}
