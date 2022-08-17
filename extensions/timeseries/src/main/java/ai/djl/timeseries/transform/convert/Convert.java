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

package ai.djl.timeseries.transform.convert;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;

import java.util.List;

/** this is a class use to convert the shape of {@link NDArray} in {@link TimeSeriesData}. */
public final class Convert {

    private Convert() {}

    /**
     * Stack fields together using {@link NDArrays#concat(NDList)}. when hStack = false, axis = 0
     * Otherwise axis = 1.
     *
     * @param outputField Field names to use for the output
     * @param inputFields Fields to stack together
     * @param dropInputs If set to true the input fields will be dropped
     * @param hStack To stack horizontally instead of vertically
     * @param data the {@link TimeSeriesData} to operate on
     */
    public static void vstackFeatures(
            FieldName outputField,
            List<FieldName> inputFields,
            boolean dropInputs,
            boolean hStack,
            TimeSeriesData data) {
        NDList ndList = new NDList();
        for (FieldName fieldName : inputFields) {
            NDArray temp = data.get(fieldName);
            if (temp != null) {
                ndList.add(data.get(fieldName));
            }
        }

        NDArray output = NDArrays.concat(ndList, hStack ? 1 : 0);
        data.setField(outputField, output);
        if (dropInputs) {
            for (FieldName fieldName : inputFields) {
                if (fieldName != outputField) {
                    data.remove(fieldName);
                }
            }
        }
    }

    /**
     * Stack fields together using {@link NDArrays#concat(NDList)}.
     *
     * <p>set hStack = false, dropInputs = true
     *
     * @param outputField Field names to use for the output
     * @param inputFields Fields to stack together
     * @param data the {@link TimeSeriesData} to operate on
     */
    public static void vstackFeatures(
            FieldName outputField, List<FieldName> inputFields, TimeSeriesData data) {
        vstackFeatures(outputField, inputFields, true, false, data);
    }
}
