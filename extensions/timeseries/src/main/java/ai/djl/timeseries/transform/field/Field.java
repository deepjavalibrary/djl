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

package ai.djl.timeseries.transform.field;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;

import java.util.ArrayList;
import java.util.List;

/** this is a class use to operate field name in {@link TimeSeriesData}. */
public final class Field {

    private Field() {}

    /**
     * Remove fields names if present.
     *
     * @param manager default {@link NDManager}
     * @param fieldNames List of names of the fields that will be removed
     * @param data the {@link TimeSeriesData} to operate on
     * @return the result {@link TimeSeriesData}
     */
    public static TimeSeriesData removeFields(
            NDManager manager, List<FieldName> fieldNames, TimeSeriesData data) {
        for (FieldName k : fieldNames) {
            data.remove(k);
        }
        return data;
    }

    /**
     * Sets a field in the dictionary with the given value.
     *
     * @param manager default {@link NDManager}
     * @param outputField Name of the field that will be set
     * @param value Value to be set
     * @param data the {@link TimeSeriesData} to operate on
     * @return the result {@link TimeSeriesData}
     */
    public static TimeSeriesData setField(
            NDManager manager, FieldName outputField, NDArray value, TimeSeriesData data) {
        data.remove(outputField);
        data.add(outputField, value);
        return data;
    }

    /**
     * Only keep the listed fields.
     *
     * @param manager default {@link NDManager}
     * @param inputFields List of fields to keep
     * @param data the {@link TimeSeriesData} to operate on
     * @return the result {@link TimeSeriesData}
     */
    public static TimeSeriesData selectField(
            NDManager manager, List<String> inputFields, TimeSeriesData data) {
        List<NDArray> values = new ArrayList<>();
        for (String field : inputFields) {
            values.add(data.get(field));
        }
        List<String> keys = new ArrayList<>(inputFields);
        return new TimeSeriesData(keys, values);
    }
}
