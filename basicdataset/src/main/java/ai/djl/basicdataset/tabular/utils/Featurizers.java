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
package ai.djl.basicdataset.tabular.utils;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

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
        return getStringFeaturizer(true);
    }

    /**
     * Returns the default String {@link Featurizer}.
     *
     * @param onehotEncode true to use onehot encoding
     * @return the default String {@link Featurizer}
     */
    public static Featurizer getStringFeaturizer(boolean onehotEncode) {
        if (onehotEncode) {
            return new PreparedOneHotStringFeaturizer();
        } else {
            return STRING_FEATURIZER;
        }
    }

    /**
     * Returns a new instance of String {@link Featurizer}.
     *
     * @param map a map contains categorical value maps to index
     * @param onehotEncode true to use onehot encoding
     * @return a new instance of String {@link Featurizer}
     */
    public static Featurizer getStringFeaturizer(Map<String, Integer> map, boolean onehotEncode) {
        if (onehotEncode) {
            return new OneHotStringFeaturizer(map);
        } else {
            return new StringFeaturizer(map);
        }
    }

    private static final class NumericFeaturizer implements Featurizer {

        /** {@inheritDoc} */
        @Override
        public void featurize(DynamicBuffer buf, String input) {
            buf.put(Float.parseFloat(input));
        }
    }

    private static class OneHotStringFeaturizer implements Featurizer {
        protected Map<String, Integer> map;

        public OneHotStringFeaturizer(Map<String, Integer> map) {
            this.map = map;
        }

        /** {@inheritDoc} */
        @Override
        public void featurize(DynamicBuffer buf, String input) {
            for (int i = 0; i < map.size(); ++i) {
                buf.put(i == map.get(input) ? 1 : 0);
            }
        }
    }

    private static final class PreparedOneHotStringFeaturizer extends OneHotStringFeaturizer
            implements PreparedFeaturizer {

        public PreparedOneHotStringFeaturizer() {
            super(null);
        }

        /** {@inheritDoc} */
        @Override
        public void prepare(List<String> inputs) {
            map = new ConcurrentHashMap<>();
            for (String input : inputs) {
                if (!map.containsKey(input)) {
                    map.put(input, map.size());
                }
            }
        }
    }

    private static final class StringFeaturizer implements Featurizer {

        private Map<String, Integer> map;
        private boolean autoMap;

        StringFeaturizer() {
            this.map = new HashMap<>();
            this.autoMap = true;
        }

        StringFeaturizer(Map<String, Integer> map) {
            this.map = map;
        }

        /** {@inheritDoc} */
        @Override
        public void featurize(DynamicBuffer buf, String input) {
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

    /**
     * A featurizer implemented for feature of date type using epoch day (number of days since
     * 1970-01-01).
     */
    public static final class EpochDayFeaturizer implements Featurizer {

        String datePattern;

        /**
         * Constructs a {@link EpochDayFeaturizer}.
         *
         * @param datePattern the pattern that dates are found in the data table column
         */
        public EpochDayFeaturizer(String datePattern) {
            this.datePattern = datePattern;
        }

        /**
         * Featurize the feature of date type to epoch day (the number of days passed since
         * 1970-01-01) and put it into float buffer, so that it can be used for future training in a
         * simple way.
         *
         * @param buf the float buffer to be filled
         * @param input the date string in the format {@code yyyy-MM-dd}
         */
        @Override
        public void featurize(DynamicBuffer buf, String input) {
            LocalDate ld = LocalDate.parse(input, DateTimeFormatter.ofPattern(datePattern));
            long day = ld.toEpochDay();
            buf.put(day);
        }
    }
}
