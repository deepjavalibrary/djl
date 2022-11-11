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
import ai.djl.ndarray.types.DataType;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;

/** A class used to convert the shape of {@link NDArray} in {@link TimeSeriesData}. */
public final class Convert {

    private Convert() {}

    /**
     * Stacks fields together using {@link NDArrays#concat(NDList)}. when hStack = false, axis = 0
     * Otherwise axis = 1.
     *
     * @param outputField the field names to use for the output
     * @param inputFields the fields to stack together
     * @param dropInputs the input fields will be dropped if set to true
     * @param hStack to stack horizontally instead of vertically
     * @param data the {@link TimeSeriesData} to operate on
     */
    public static void vstackFeatures(
            FieldName outputField,
            FieldName[] inputFields,
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
     * Stacks fields together using {@link NDArrays#concat(NDList)}.
     *
     * <p>set hStack = false, dropInputs = true
     *
     * @param outputField the field names to use for the output
     * @param inputFields the fields to stack together
     * @param data the {@link TimeSeriesData} to operate on
     */
    public static void vstackFeatures(
            FieldName outputField, FieldName[] inputFields, TimeSeriesData data) {
        vstackFeatures(outputField, inputFields, true, false, data);
    }

    /**
     * Converts the data type of {@link NDArray} and check its dimension.
     *
     * @param field aim field
     * @param expectedDim expected number of dimensions
     * @param dataType {@link DataType} to use
     * @param data the {@link TimeSeriesData} to operate on
     */
    public static void asArray(
            FieldName field, int expectedDim, DataType dataType, TimeSeriesData data) {
        NDArray value = data.get(field);
        if (value == null) {
            throw new IllegalArgumentException(String.format("%s don't map to any NDArray", field));
        }

        value = value.toType(dataType, true);
        if (value.getShape().dimension() != expectedDim) {
            throw new IllegalArgumentException(
                    String.format(
                            "Input for field \"%s\" does not have the required dimension (field:"
                                    + " %s, dim: %d, expected: %d)",
                            field, field, value.getShape().dimension(), expectedDim));
        }
        data.setField(field, value);
    }

    /**
     * Converts the data type of {@link NDArray} and check its dimension.
     *
     * @param field aim field
     * @param expectedDim expected number of dimensions
     * @param data the {@link TimeSeriesData} to operate on
     */
    public static void asArray(FieldName field, int expectedDim, TimeSeriesData data) {
        asArray(field, expectedDim, DataType.FLOAT32, data);
    }
}
