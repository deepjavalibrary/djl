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

import ai.djl.basicdataset.tabular.utils.DynamicBuffer;
import ai.djl.basicdataset.tabular.utils.Featurizer;

import java.time.LocalDateTime;
import java.time.ZoneOffset;

/**
 * An interface that convert String to {@link LocalDateTime} as the start field of {@link
 * ai.djl.timeseries.TimeSeriesData}.
 */
public interface TimeFeaturizer extends Featurizer {

    /** {@inheritDoc} */
    @Override
    default void featurize(DynamicBuffer buf, String input) {
        throw new IllegalArgumentException(
                "Please use the other featurize for DateTimeFeaturizers");
    }

    /**
     * Return the parsed time data.
     *
     * @param input the string input
     * @return the parsed {@link LocalDateTime}
     */
    LocalDateTime featurize(String input);

    /** {@inheritDoc} */
    @Override
    default int dataRequired() {
        return 2;
    }

    /** {@inheritDoc} */
    @Override
    default Object deFeaturize(float[] data) {
        return LocalDateTime.ofEpochSecond((long) data[0], (int) data[1], ZoneOffset.UTC);
    }
}
