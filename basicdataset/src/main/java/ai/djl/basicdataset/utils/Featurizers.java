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
package ai.djl.basicdataset.utils;

import java.util.HashMap;
import java.util.Map;

/** A utility class provides helper functions to create {@link Featurizer}. */
public final class Featurizers {

    private static final Featurizer NUMERIC_FEATURIZER = new NumericFeaturizer();
    private static final Featurizer STRING_FEATURIZER = new StringFeaturizer();

    private Featurizers() {}

    /**
     * Returns the default numeric {@link Featurizer}.
     *
     * @return the default numeric {@link Featurizer}
     */
    public static Featurizer getNumericFeaturizer() {
        return NUMERIC_FEATURIZER;
    }

    /**
     * Returns the default String {@link Featurizer}.
     *
     * @return the default String {@link Featurizer}
     */
    public static Featurizer getStringFeaturizer() {
        return STRING_FEATURIZER;
    }

    /**
     * Returns a new instance of String {@link Featurizer}.
     *
     * @param map a map contains categorical value maps to index
     * @param onehotEncode true if use onehot encode
     * @return a new instance of String {@link Featurizer}
     */
    public static Featurizer getStringFeaturizer(Map<String, Integer> map, boolean onehotEncode) {
        return new StringFeaturizer(map, onehotEncode);
    }

    private static final class NumericFeaturizer implements Featurizer {

        /** {@inheritDoc} */
        @Override
        public void featurize(DynamicBuffer buf, String input) {
            buf.put(Float.parseFloat(input));
        }
    }

    private static final class StringFeaturizer implements Featurizer {

        private Map<String, Integer> map;
        private boolean onehotEncode;
        private boolean autoMap;

        StringFeaturizer() {
            this.map = new HashMap<>();
            this.autoMap = true;
        }

        StringFeaturizer(Map<String, Integer> map, boolean onehotEncode) {
            this.map = map;
            this.onehotEncode = onehotEncode;
        }

        /** {@inheritDoc} */
        @Override
        public void featurize(DynamicBuffer buf, String input) {
            if (onehotEncode) {
                for (int i = 0; i < map.size(); ++i) {
                    buf.put(i == map.get(input) ? 1 : 0);
                }
                return;
            }

            Integer index = map.get(input);
            if (index != null) {
                buf.put(index);
                return;
            }

            if (!autoMap) {
                throw new IllegalArgumentException("Value: " + input + " not found in the map.");
            }
            int value = map.size();
            map.put(input, value);
            buf.put(value);
        }
    }
}
