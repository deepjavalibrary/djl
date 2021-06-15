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
package ai.djl.examples.inference.biggan;

public final class BigGANInput {
    private int sampleSize;
    private float truncation;
    private BigGANCategory category;

    public BigGANInput(int sampleSize, float truncation, BigGANCategory category) {
        this.sampleSize = sampleSize;
        this.truncation = truncation;
        this.category = category;
    }

    BigGANInput(Builder builder) {
        this.sampleSize = builder.sampleSize;
        this.truncation = builder.truncation;
        this.category = builder.category;
    }

    public int getSampleSize() {
        return sampleSize;
    }

    public float getTruncation() {
        return truncation;
    }

    public BigGANCategory getCategory() {
        return category;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private int sampleSize = 5;
        private float truncation = 0.5f;
        private BigGANCategory category;

        Builder() {
            category = BigGANCategory.of("Egyptian cat");
        }

        public Builder optSampleSize(int sampleSize) {
            this.sampleSize = sampleSize;
            return this;
        }

        public Builder optTruncation(float truncation) {
            this.truncation = truncation;
            return this;
        }

        public Builder setCategory(BigGANCategory category) {
            this.category = category;
            return this;
        }

        public BigGANInput build() {
            return new BigGANInput(this);
        }
    }
}
