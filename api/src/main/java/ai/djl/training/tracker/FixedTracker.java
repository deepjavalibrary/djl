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
package ai.djl.training.tracker;

/**
 * {@link FixedTracker} is an implementation of {@link Tracker} which returns a fixed value.
 *
 * @see Tracker
 */
class FixedTracker implements Tracker {

    private float value;

    /**
     * Creates a new instance of {@link FixedTracker}.
     *
     * @param builder the builder used to build this object
     */
    public FixedTracker(Builder builder) {
        this.value = builder.value;
    }

    /** {@inheritDoc} */
    @Override
    public float getNewValue(int numUpdate) {
        return value;
    }

    /**
     * Creates a builder to build a {@link FixedTracker}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct an {@link FixedTracker} object. */
    public static final class Builder {

        private float value;

        private Builder() {}

        public Builder setValue(float value) {
            this.value = value;
            return this;
        }

        /**
         * Builds a {@link FixedTracker} block.
         *
         * @return the {@link FixedTracker} block
         */
        public FixedTracker build() {
            return new FixedTracker(this);
        }
    }
}
