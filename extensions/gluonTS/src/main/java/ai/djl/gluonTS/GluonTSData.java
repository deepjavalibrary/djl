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
package ai.djl.gluonTS;

import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * {@link GluonTSData} is a DataEntry for managing GluonTS data in preprocess. It contains a
 * key-values entries mapping the {@link FieldName} and {@link NDArray} to generate the time
 * features.
 *
 * <p>This class provides a convenient way for user to featurize the data.
 */
public class GluonTSData extends PairList<String, NDArray> {

    private LocalDateTime startTime;
    private LocalDateTime foreCastStartTime;

    public GluonTSData() {
        super();
    }

    public GluonTSData(int initialCapacity) {
        super(initialCapacity);
    }

    public GluonTSData(List<String> keys, List<NDArray> values) {
        super(keys, values);
    }

    public GluonTSData(List<Pair<String, NDArray>> list) {
        super(list);
    }

    public GluonTSData(Map<String, NDArray> map) {
        super(map);
    }

    /**
     * Constructs a {@link NDList} containing the remaining {@link NDArray} for {@link FieldName}.
     *
     * @return a {@link NDList}
     */
    public NDList toNDList() {
        List<NDArray> arrays = this.values();
        for (int i = 0; i < arrays.size(); i++) {
            arrays.get(i).setName("data" + i);
        }
        return new NDList(this.values());
    }

    /**
     * Get the time series start time.
     *
     * @return a {@link LocalDateTime} representing start time;
     */
    public LocalDateTime getStartTime() {
        return startTime;
    }

    /**
     * Get the time series forecasting time.
     *
     * @return a {@link LocalDateTime} representing the time to forecast.
     */
    public LocalDateTime getForeCastStartTime() {
        return foreCastStartTime;
    }

    /**
     * Adds a fieldName and value to the list.
     *
     * @param fieldName the {@link FieldName}.
     * @param value the {@link NDArray} value
     */
    public void add(FieldName fieldName, NDArray value) {
        add(fieldName.toString(), value);
    }

    /**
     * Returns the value for the fieldName
     *
     * @param fieldName the {@link FieldName} of the element to get.
     * @return the {@link NDArray} value for the {@link FieldName}.
     */
    public NDArray get(FieldName fieldName) {
        return get(fieldName.toString());
    }

    /**
     * Replace the existing {@link NDArray} of {@link FieldName} to the value.
     *
     * @param fieldName the {@link FieldName}.
     * @param value the {@link NDArray} value.
     */
    public void setField(String fieldName, NDArray value) {
        remove(fieldName);
        add(fieldName, value);
    }

    /**
     * Replace the existing {@link NDArray} of {@link FieldName} to the value.
     *
     * @param fieldName the {@link FieldName}.
     * @param value the {@link NDArray} value.
     */
    public void setField(FieldName fieldName, NDArray value) {
        setField(fieldName.name(), value);
    }

    /**
     * Set the time series start time.
     *
     * @param value the {@link LocalDateTime} start time.
     */
    public void setStartTime(LocalDateTime value) {
        this.startTime = value;
    }

    /**
     * Set the time series forecasting time.
     *
     * @param value the {@link LocalDateTime} time to forecast
     */
    public void setForeCastStartTime(LocalDateTime value) {
        this.foreCastStartTime = value;
    }

    /**
     * Remove the key-value pair for the {@link FieldName}
     *
     * @param fieldName the {@link FieldName} to be removed.
     */
    public void remove(FieldName fieldName) {
        remove(fieldName.toString());
    }
}
