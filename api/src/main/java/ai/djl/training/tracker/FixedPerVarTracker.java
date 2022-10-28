/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@link FixedPerVarTracker} is an implementation of {@link Tracker} which returns a fixed value.
 *
 * @see Tracker
 */
public class FixedPerVarTracker implements ParameterTracker {

    private float value;
    private Map<String, Float> valueMap;

    /**
     * Creates a new instance of {@link FixedPerVarTracker}.
     *
     * @param builder the builder used to build this object
     */
    public FixedPerVarTracker(Builder builder) {
        this.value = builder.value;
        this.valueMap = builder.valueMap;
    }

    /** {@inheritDoc} */
    @Override
    public float getNewValue(String parameterId, int numUpdate) {
        return valueMap.getOrDefault(parameterId, this.value);
    }

    /**
     * Creates a builder to build a {@link FixedPerVarTracker}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct an {@link FixedPerVarTracker} object. */
    public static final class Builder {

        private float value;
        private Map<String, Float> valueMap = new ConcurrentHashMap<>();

        /** Create a builder for {@link FixedPerVarTracker}. */
        private Builder() {}

        /**
         * Set the default learning rate.
         *
         * @param value the default learning rate
         * @return builder
         */
        public Builder setDefaultValue(float value) {
            this.value = value;
            return this;
        }

        /**
         * Add a kv pair of parameter and its learning rate.
         *
         * @param parameterId the parameter id
         * @param value the default learning rate
         * @return builder
         */
        public Builder put(String parameterId, float value) {
            this.valueMap.put(parameterId, value);
            return this;
        }

        /**
         * Add kv pairs of parameter and its learning rate.
         *
         * @param valueMap stores parameterId and learning rate
         * @return builder
         */
        public Builder putAll(Map<String, Float> valueMap) {
            this.valueMap.putAll(valueMap);
            return this;
        }

        /**
         * Builds a {@link FixedPerVarTracker} block.
         *
         * @return the {@link FixedPerVarTracker} block
         */
        public FixedPerVarTracker build() {
            return new FixedPerVarTracker(this);
        }
    }
}
