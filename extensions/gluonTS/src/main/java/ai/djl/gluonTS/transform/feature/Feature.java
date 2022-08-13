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

package ai.djl.gluonTS.transform.feature;

import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

import java.time.Duration;
import java.time.LocalDateTime;
import java.time.Period;
import java.time.temporal.TemporalAmount;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

/** this is a class use to add feature in {@link GluonTSData} */
public class Feature {

    public Feature() {}

    /**
     * Replaces missing values in a {@link NDArray} (NaNs) with a dummy value and adds an
     * "observed"-indicator that is "1" when values are observed and "0" when values are missing
     *
     * @param manager default {@link NDManager}
     * @param targetField Field for which missing values will be replaced
     * @param outputField Field name to use for the indicator
     * @param data the {@link GluonTSData} to operate on
     * @return the result {@link GluonTSData}
     */
    public static GluonTSData addObservedValuesIndicator(
            NDManager manager, FieldName targetField, FieldName outputField, GluonTSData data) {
        NDArray value = data.get(targetField);
        data.setField(targetField, dummyValueImputation(manager, value, 0f));
        NDArray nanEntries = value.isNaN();
        data.setField(outputField, nanEntries.logicalNot().toType(value.getDataType(), false));
        return data;
    }

    /**
     * Adds a set of time features.
     *
     * @param manager default {@link NDManager}
     * @param startField Field with the start time stamp of the time series
     * @param targetField Field with the array containing the time series values
     * @param outputField Field name for result.
     * @param timeFeatures list of time features to use.
     * @param predictionLength Prediction length
     * @param freq Prediction time frequency
     * @param data the {@link GluonTSData} to operate on
     * @return the result {@link GluonTSData}
     */
    public static GluonTSData addTimeFeature(
            NDManager manager,
            FieldName startField,
            FieldName targetField,
            FieldName outputField,
            List<BiFunction<NDManager, List<LocalDateTime>, NDArray>> timeFeatures,
            int predictionLength,
            String freq,
            GluonTSData data) {
        if (timeFeatures.size() == 0) {
            data.setField(outputField, null);
        }

        LocalDateTime start = data.getStartTime();
        int length = targetTransformationLength(data.get(targetField), predictionLength, false);

        String formattedFreq = freq;
        if (!freq.matches("\\d+.*")) {
            formattedFreq = "1" + formattedFreq;
        }
        formattedFreq = "P" + formattedFreq;

        TemporalAmount timeFreq;
        if (freq.endsWith("H") || freq.endsWith("T") || freq.endsWith("S")) {
            timeFreq = Duration.parse(formattedFreq);
        } else {
            timeFreq = Period.parse(formattedFreq);
        }

        List<LocalDateTime> index =
                new ArrayList<LocalDateTime>() {
                    {
                        LocalDateTime temp = start;
                        for (int i = 0; i < length; i++) {
                            add(temp);
                            temp = temp.plus(timeFreq);
                        }
                    }
                };

        NDList outputs =
                new NDList(timeFeatures.size()) {
                    {
                        for (BiFunction<NDManager, List<LocalDateTime>, NDArray> f : timeFeatures) {
                            add(f.apply(manager, index));
                        }
                    }
                };
        data.setField(outputField, NDArrays.stack(outputs));
        return data;
    }

    /**
     * Adds on 'age' feature to the {@link GluonTSData}
     *
     * <p>The age feature starts with a small value at the start of the time series and grows over
     * time.
     *
     * @param manager default {@link NDManager}
     * @param targetField Field with target values (array) of time series
     * @param outputField Field name to use for the output.
     * @param predictionLength Prediction length
     * @param logScale If set to true the age feature grows logarithmically otherwise linearly over
     *     time.
     * @param data the {@link GluonTSData} to operate on
     * @return the result {@link GluonTSData}
     */
    public static GluonTSData addAgeFeature(
            NDManager manager,
            FieldName targetField,
            FieldName outputField,
            int predictionLength,
            boolean logScale,
            GluonTSData data) {

        NDArray targetData = data.get(targetField);
        int length = targetTransformationLength(targetData, predictionLength, false);

        NDArray age = manager.arange(0, length, 1, targetData.getDataType());
        if (logScale) {
            age = age.add(2f).log10();
        }
        age = age.reshape(new Shape(1, length));

        data.setField(outputField, age);
        return data;
    }

    /**
     * Adds on 'age' feature to the {@link GluonTSData}, set logScale = ture
     *
     * <p>The age feature starts with a small value at the start of the time series and grows over
     * time.
     *
     * @param manager default {@link NDManager}
     * @param targetField Field with target values (array) of time series
     * @param outputField Field name to use for the output.
     * @param predictionLength Prediction length
     * @param data the {@link GluonTSData} to operate on
     * @return the result {@link GluonTSData}
     */
    public static GluonTSData addAgeFeature(
            NDManager manager,
            FieldName targetField,
            FieldName outputField,
            int predictionLength,
            GluonTSData data) {
        return addAgeFeature(manager, targetField, outputField, predictionLength, true, data);
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
