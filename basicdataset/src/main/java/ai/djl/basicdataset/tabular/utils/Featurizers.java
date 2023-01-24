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

import ai.djl.modality.Classifications;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;

/** A utility class provides helper functions to create {@link Featurizer}. */
public final class Featurizers {

    private static final Featurizer NUMERIC_FEATURIZER = new NumericFeaturizer();

    private Featurizers() {}

    /**
     * Returns the default numeric {@link Featurizer}.
     *
     * @return the default numeric {@link Featurizer}
     */
    public static Featurizer getNumericFeaturizer() {
        return getNumericFeaturizer(false);
    }

    /**
     * Returns the default numeric {@link Featurizer}.
     *
     * @param normalize true to normalize (with mean and std) the values
     * @return the default numeric {@link Featurizer}
     */
    public static Featurizer getNumericFeaturizer(boolean normalize) {
        if (normalize) {
            return new NormalizedNumericFeaturizer();
        } else {
            return NUMERIC_FEATURIZER;
        }
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
            return new StringFeaturizer();
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

    /**
     * Constructs an {@link EpochDayFeaturizer} for representing dates using the epoch day (number
     * of days since 1970-01-01).
     *
     * @param datePattern the pattern that dates are found in the data table column
     * @return a new instance of {@link EpochDayFeaturizer}
     */
    public static Featurizer getEpochDayFeaturizer(String datePattern) {
        return new EpochDayFeaturizer(datePattern);
    }

    private static final class NumericFeaturizer implements Featurizer {

        /** {@inheritDoc} */
        @Override
        public void featurize(DynamicBuffer buf, String input) {
            buf.put(Float.parseFloat(input));
        }

        /** {@inheritDoc} */
        @Override
        public int dataRequired() {
            return 1;
        }

        /** {@inheritDoc} */
        @Override
        public Object deFeaturize(float[] data) {
            return data[0];
        }
    }

    private static final class NormalizedNumericFeaturizer implements PreparedFeaturizer {

        private float mean;
        private float std;

        /** {@inheritDoc} */
        @Override
        public void featurize(DynamicBuffer buf, String input) {
            float value = (Float.parseFloat(input) - mean) / std;
            buf.put(value);
        }

        /** {@inheritDoc} */
        @Override
        public void prepare(List<String> inputs) {
            calculateMean(inputs);
            calculateStd(inputs);
        }

        private void calculateMean(List<String> inputs) {
            double sum = 0;
            for (String input : inputs) {
                sum += Float.parseFloat(input);
            }
            mean = (float) (sum / inputs.size());
        }

        private void calculateStd(List<String> inputs) {
            double sum = 0;
            for (String input : inputs) {
                sum += Math.pow(Float.parseFloat(input) - mean, 2);
            }
            std = (float) Math.sqrt(sum / inputs.size());
        }

        /** {@inheritDoc} */
        @Override
        public int dataRequired() {
            return 1;
        }

        /** {@inheritDoc} */
        @Override
        public Object deFeaturize(float[] data) {
            return data[0];
        }
    }

    private abstract static class BaseStringFeaturizer implements Featurizer {
        protected Map<String, Integer> map;
        protected List<String> classNames;

        @SuppressWarnings("PMD.ConstructorCallsOverridableMethod")
        public BaseStringFeaturizer(Map<String, Integer> map) {
            this.map = map;
            if (map != null) {
                buildClassNames();
            }
        }

        @Override
        public int dataRequired() {
            return map.size();
        }

        @Override
        public Object deFeaturize(float[] data) {
            List<Double> probabilities = new ArrayList<>(data.length);
            for (Float d : data) {
                probabilities.add((double) d);
            }
            return new Classifications(classNames, probabilities);
        }

        protected void buildClassNames() {
            classNames = Arrays.asList(new String[map.size()]);
            for (Map.Entry<String, Integer> entry : map.entrySet()) {
                classNames.set(entry.getValue(), entry.getKey());
            }
        }
    }

    private static class OneHotStringFeaturizer extends BaseStringFeaturizer {

        public OneHotStringFeaturizer(Map<String, Integer> map) {
            super(map);
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
            TreeSet<String> uniqueInputs = new TreeSet<>(inputs);
            for (String input : uniqueInputs) {
                if (!map.containsKey(input)) {
                    map.put(input, map.size());
                }
            }
            buildClassNames();
        }
    }

    private static final class StringFeaturizer extends BaseStringFeaturizer {

        private boolean autoMap;

        StringFeaturizer() {
            super(new HashMap<>());
            this.autoMap = true;
        }

        StringFeaturizer(Map<String, Integer> map) {
            super(map);
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

        @Override
        public Object deFeaturize(float[] data) {
            if (classNames.size() != map.size()) {
                // May have to rebuild class names first if new ones were added
                buildClassNames();
            }

            return super.deFeaturize(data);
        }
    }

    /**
     * A featurizer implemented for feature of date type using epoch day (number of days since
     * 1970-01-01).
     */
    private static final class EpochDayFeaturizer implements Featurizer {

        String datePattern;

        /**
         * Constructs a {@link EpochDayFeaturizer}.
         *
         * @param datePattern the pattern that dates are found in the data table column
         */
        EpochDayFeaturizer(String datePattern) {
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

        @Override
        public int dataRequired() {
            return 1;
        }

        @Override
        public Object deFeaturize(float[] data) {
            return LocalDate.ofEpochDay(Math.round(data[0]));
        }
    }
}
