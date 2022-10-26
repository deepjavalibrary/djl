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

package ai.djl.timeseries.transform.feature;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;

import java.time.Duration;
import java.time.LocalDateTime;
import java.time.Period;
import java.time.temporal.TemporalAmount;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

/** this is a class use to add feature in {@link TimeSeriesData}. */
public final class Feature {

    private Feature() {}

    /**
     * Replaces missing values in a {@link NDArray} (NaNs) with a dummy value and adds an
     * "observed"-indicator that is "1" when values are observed and "0" when values are missing.
     *
     * @param manager default {@link NDManager}
     * @param targetField Field for which missing values will be replaced
     * @param outputField Field name to use for the indicator
     * @param data the {@link TimeSeriesData} to operate on
     */
    public static void addObservedValuesIndicator(
            NDManager manager, FieldName targetField, FieldName outputField, TimeSeriesData data) {
        NDArray value = data.get(targetField);
        data.setField(targetField, dummyValueImputation(manager, value, 0f));
        NDArray nanEntries = value.isNaN();
        data.setField(outputField, nanEntries.logicalNot().toType(value.getDataType(), false));
    }

    /**
     * Adds a set of time features.
     *
     * @param manager default {@link NDManager}
     * @param startField Field with the start time stamp of the time series
     * @param targetField Field with the array containing the time series values
     * @param outputField Field name for result
     * @param timeFeatures list of time features to use
     * @param predictionLength Prediction length
     * @param freq Prediction time frequency
     * @param data the {@link TimeSeriesData} to operate on
     */
    public static void addTimeFeature(
            NDManager manager,
            FieldName startField,
            FieldName targetField,
            FieldName outputField,
            List<BiFunction<NDManager, List<LocalDateTime>, NDArray>> timeFeatures,
            int predictionLength,
            String freq,
            TimeSeriesData data) {
        addTimeFeature(
                manager,
                startField,
                targetField,
                outputField,
                timeFeatures,
                predictionLength,
                freq,
                data,
                false);
    }

    /**
     * Adds a set of time features.
     *
     * @param manager default {@link NDManager}
     * @param startField Field with the start time stamp of the time series
     * @param targetField Field with the array containing the time series values
     * @param outputField Field name for result
     * @param timeFeatures list of time features to use
     * @param predictionLength Prediction length
     * @param freq Prediction time frequency
     * @param data the {@link TimeSeriesData} to operate on
     * @param isTrain Whether it is training
     */
    public static void addTimeFeature(
            NDManager manager,
            FieldName startField,
            FieldName targetField,
            FieldName outputField,
            List<BiFunction<NDManager, List<LocalDateTime>, NDArray>> timeFeatures,
            int predictionLength,
            String freq,
            TimeSeriesData data,
            boolean isTrain) {
        if (timeFeatures.isEmpty()) {
            data.setField(outputField, null);
        }

        LocalDateTime start = data.getStartTime();
        int length = targetTransformationLength(data.get(targetField), predictionLength, isTrain);

        StringBuilder sb = new StringBuilder();
        sb.append(freq);
        if (!freq.matches("\\d+.*")) {
            sb.insert(0, 1);
        }
        sb.insert(0, "P");
        String formattedFreq = sb.toString();

        TemporalAmount timeFreq;
        if (freq.endsWith("H") || freq.endsWith("T") || freq.endsWith("S")) {
            timeFreq = Duration.parse(formattedFreq);
        } else {
            timeFreq = Period.parse(formattedFreq);
        }

        List<LocalDateTime> index = new ArrayList<>();
        LocalDateTime temp = start;
        for (int i = 0; i < length; i++) {
            index.add(temp);
            temp = temp.plus(timeFreq);
        }

        NDList outputs = new NDList(timeFeatures.size());
        for (BiFunction<NDManager, List<LocalDateTime>, NDArray> f : timeFeatures) {
            outputs.add(f.apply(manager, index));
        }

        data.setField(outputField, NDArrays.stack(outputs));
    }

    /**
     * Adds on 'age' feature to the {@link TimeSeriesData}.
     *
     * <p>The age feature starts with a small value at the start of the time series and grows over
     * time.
     *
     * @param manager default {@link NDManager}
     * @param targetField Field with target values (array) of time series
     * @param outputField Field name to use for the output
     * @param predictionLength Prediction length
     * @param logScale If set to true the age feature grows logarithmically otherwise linearly over
     *     time.
     * @param data the {@link TimeSeriesData} to operate on
     * @return the result {@link TimeSeriesData}
     */
    public static TimeSeriesData addAgeFeature(
            NDManager manager,
            FieldName targetField,
            FieldName outputField,
            int predictionLength,
            boolean logScale,
            TimeSeriesData data) {
        return addAgeFeature(
                manager, targetField, outputField, predictionLength, logScale, data, false);
    }

    /**
     * Adds on 'age' feature to the {@link TimeSeriesData}.
     *
     * <p>The age feature starts with a small value at the start of the time series and grows over
     * time.
     *
     * @param manager default {@link NDManager}
     * @param targetField Field with target values (array) of time series
     * @param outputField Field name to use for the output
     * @param predictionLength Prediction length
     * @param logScale If set to true the age feature grows logarithmically otherwise linearly over
     *     time.
     * @param data the {@link TimeSeriesData} to operate on
     * @param isTrain Whether it is training
     * @return the result {@link TimeSeriesData}
     */
    public static TimeSeriesData addAgeFeature(
            NDManager manager,
            FieldName targetField,
            FieldName outputField,
            int predictionLength,
            boolean logScale,
            TimeSeriesData data,
            boolean isTrain) {

        NDArray targetData = data.get(targetField);
        int length = targetTransformationLength(targetData, predictionLength, isTrain);

        NDArray age = manager.arange(0, length, 1, targetData.getDataType());
        if (logScale) {
            age = age.add(2f).log10();
        }
        age = age.reshape(new Shape(1, length));

        data.setField(outputField, age);
        return data;
    }

    /**
     * Adds on 'age' feature to the {@link TimeSeriesData}, set logScale = ture
     *
     * <p>The age feature starts with a small value at the start of the time series and grows over
     * time.
     *
     * @param manager default {@link NDManager}
     * @param targetField Field with target values (array) of time series
     * @param outputField Field name to use for the output
     * @param predictionLength Prediction length
     * @param data the {@link TimeSeriesData} to operate on
     */
    public static void addAgeFeature(
            NDManager manager,
            FieldName targetField,
            FieldName outputField,
            int predictionLength,
            TimeSeriesData data) {
        addAgeFeature(manager, targetField, outputField, predictionLength, true, data);
    }

    private static int targetTransformationLength(
            NDArray target, int predictionLength, boolean isTrain) {
        return (int) target.getShape().tail() + (isTrain ? 0 : predictionLength);
    }

    private static NDArray dummyValueImputation(
            NDManager manager, NDArray value, float dummyValue) {
        NDArray dummyArray = manager.full(value.getShape(), dummyValue);
        return NDArrays.where(value.isNaN(), dummyArray, value);
    }
}
