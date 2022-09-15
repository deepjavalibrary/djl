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

package ai.djl.timeseries.dataset;

import ai.djl.basicdataset.tabular.utils.Featurizer;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/** A utility class provides helper functions to create {@link TimeFeaturizer}. */
public final class TimeFeaturizers {

    private TimeFeaturizers() {}

    /**
     * Construct a {@link PatternTimeFeaturizer}.
     *
     * @param datePattern the pattern that dates are found in the data table column
     * @return a new instance of {@link PatternTimeFeaturizer}
     */
    public static Featurizer getPatternTimeFeaturizer(String datePattern) {
        return new PatternTimeFeaturizer(datePattern);
    }

    /**
     * Construct a {@link ConstantTimeFeaturizer}.
     *
     * @param dateTime the date time to return for all
     * @return a new instance of {@link ConstantTimeFeaturizer}
     */
    public static Featurizer getConstantTimeFeaturizer(LocalDateTime dateTime) {
        return new ConstantTimeFeaturizer(dateTime);
    }

    /** A featurizer implemented for feature of date type. */
    public static final class PatternTimeFeaturizer implements TimeFeaturizer {

        String datePattern;

        /**
         * Constructs a {@link PatternTimeFeaturizer}.
         *
         * @param datePattern the pattern that dates are found in the data table column
         */
        PatternTimeFeaturizer(String datePattern) {
            this.datePattern = datePattern;
        }

        /** {@inheritDoc} */
        @Override
        public LocalDateTime featurize(String input) {
            return LocalDateTime.parse(input, DateTimeFormatter.ofPattern(datePattern));
        }
    }

    /** A featurizer always return a constant date. */
    public static final class ConstantTimeFeaturizer implements TimeFeaturizer {

        LocalDateTime dateTime;

        /**
         * Constructs a {@link ConstantTimeFeaturizer}.
         *
         * @param dateTime the constant date
         */
        ConstantTimeFeaturizer(LocalDateTime dateTime) {
            this.dateTime = LocalDateTime.from(dateTime);
        }

        /** {@inheritDoc} */
        @Override
        public LocalDateTime featurize(String input) {
            return dateTime;
        }
    }
}
