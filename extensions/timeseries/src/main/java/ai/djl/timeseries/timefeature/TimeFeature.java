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
package ai.djl.timeseries.timefeature;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.util.PairList;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

/** this is a class to generate time feature by frequency. */
public final class TimeFeature {

    private static final PairList<String, List<BiFunction<NDManager, List<LocalDateTime>, NDArray>>>
            FEATURES_BY_OFFSETS = init();

    private TimeFeature() {}

    /**
     * Computes feature by seconds.
     *
     * @param manager default {@link NDManager}.
     * @param index time data
     * @return the result feature
     */
    public static NDArray secondOfMinute(NDManager manager, List<LocalDateTime> index) {
        float[] data = getFeature(index, LocalDateTime::getSecond);
        return manager.create(data).divi(59f).subi(0.5);
    }

    /**
     * Computes feature by minutes.
     *
     * @param manager default {@link NDManager}.
     * @param index time data
     * @return the result feature
     */
    public static NDArray minuteOfHour(NDManager manager, List<LocalDateTime> index) {
        float[] data = getFeature(index, LocalDateTime::getMinute);
        return manager.create(data).divi(59f).subi(0.5);
    }

    /**
     * Computes feature by hours.
     *
     * @param manager default {@link NDManager}.
     * @param index time data
     * @return the result feature
     */
    public static NDArray hourOfDay(NDManager manager, List<LocalDateTime> index) {
        float[] data = getFeature(index, LocalDateTime::getHour);
        return manager.create(data).divi(23f).subi(0.5);
    }

    /**
     * Computes feature by days of the week.
     *
     * @param manager default {@link NDManager}.
     * @param index time data
     * @return the result feature
     */
    public static NDArray dayOfWeek(NDManager manager, List<LocalDateTime> index) {
        float[] data = getFeature(index, a -> a.getDayOfWeek().ordinal());
        return manager.create(data).divi(6f).subi(0.5);
    }

    /**
     * Computes feature by days fo the month.
     *
     * @param manager default {@link NDManager}.
     * @param index time data
     * @return the result feature
     */
    public static NDArray dayOfMonth(NDManager manager, List<LocalDateTime> index) {
        float[] data = getFeature(index, LocalDateTime::getDayOfMonth);
        return manager.create(data).subi(1f).divi(30f).subi(0.5f);
    }

    /**
     * Computes feature by days of the year.
     *
     * @param manager default {@link NDManager}.
     * @param index time data
     * @return the result feature
     */
    public static NDArray dayOfYear(NDManager manager, List<LocalDateTime> index) {
        float[] data = getFeature(index, LocalDateTime::getDayOfYear);
        return manager.create(data).subi(1f).divi(365f).subi(0.5f);
    }

    /**
     * Computes feature by months of the year.
     *
     * @param manager default {@link NDManager}.
     * @param index time data
     * @return the result feature
     */
    public static NDArray monthOfYear(NDManager manager, List<LocalDateTime> index) {
        float[] data = getFeature(index, LocalDateTime::getMonthValue);
        return manager.create(data).subi(1f).divi(11f).subi(0.5);
    }

    /**
     * Computes feature by weeks of the year.
     *
     * @param manager default {@link NDManager}.
     * @param index time data
     * @return the result feature
     */
    public static NDArray weekOfYear(NDManager manager, List<LocalDateTime> index) {
        throw new UnsupportedOperationException("weekOfYear is not supported yet");
    }

    private static float[] getFeature(
            List<LocalDateTime> index, Function<LocalDateTime, Number> function) {
        float[] data = new float[index.size()];
        int i = 0;
        for (LocalDateTime time : index) {
            data[i++] = function.apply(time).floatValue();
        }
        return data;
    }

    /**
     * Returns a list of time features that will be appropriate for the given frequency string.
     *
     * @param freqStr Frequency string of the form [multiple][granularity] such as "12H", "1D"
     * @return time features
     */
    public static List<BiFunction<NDManager, List<LocalDateTime>, NDArray>> timeFeaturesFromFreqStr(
            String freqStr) {
        return FEATURES_BY_OFFSETS.get(freqStr);
    }

    private static PairList<String, List<BiFunction<NDManager, List<LocalDateTime>, NDArray>>>
            init() {
        List<String> keys = Arrays.asList("Y", "Q", "M", "W", "D", "H", "T", "S");
        List<List<BiFunction<NDManager, List<LocalDateTime>, NDArray>>> values =
                Arrays.asList(
                        Collections.emptyList(),
                        Collections.singletonList(TimeFeature::monthOfYear),
                        Collections.singletonList(TimeFeature::monthOfYear),
                        Arrays.asList(TimeFeature::dayOfMonth, TimeFeature::weekOfYear),
                        Arrays.asList(
                                TimeFeature::dayOfWeek,
                                TimeFeature::dayOfMonth,
                                TimeFeature::dayOfYear),
                        Arrays.asList(
                                TimeFeature::hourOfDay,
                                TimeFeature::dayOfWeek,
                                TimeFeature::dayOfMonth,
                                TimeFeature::dayOfYear),
                        Arrays.asList(
                                TimeFeature::minuteOfHour,
                                TimeFeature::hourOfDay,
                                TimeFeature::dayOfWeek,
                                TimeFeature::dayOfMonth,
                                TimeFeature::dayOfYear),
                        Arrays.asList(
                                TimeFeature::secondOfMinute,
                                TimeFeature::minuteOfHour,
                                TimeFeature::hourOfDay,
                                TimeFeature::dayOfWeek,
                                TimeFeature::dayOfMonth,
                                TimeFeature::dayOfYear));
        return new PairList<>(keys, values);
    }
}
