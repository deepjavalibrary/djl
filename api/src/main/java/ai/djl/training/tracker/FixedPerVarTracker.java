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

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@link FixedPerVarTracker} is an implementation of {@link Tracker} which returns a fixed value.
 *
 * @see Tracker
 */
public class FixedPerVarTracker implements Tracker {

    private float value;
    private Map<String, Float> valueMap;
    private String parameterId;

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
    public float getNewValue(int numUpdate) {
        return valueMap.getOrDefault(this.parameterId, this.value);
    }

    public void setParameterId(String parameterId) {
        this.parameterId = parameterId;
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
        private Map<String, Float> valueMap = new ConcurrentHashMap<String, Float>();

        private Builder() {}

        public Builder setDefaultValue(float value) {
            this.value = value;
            return this;
        }

        public Builder put(String parameterId, float value) {
            this.valueMap.put(parameterId, value);
            return this;
        }

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
