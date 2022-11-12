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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.transform.TimeSeriesTransform;

/** Convert the data type of {@link NDArray} and check its dimension. */
public class AsArray implements TimeSeriesTransform {

    private FieldName field;
    private DataType dataType;
    private int expectedDim;

    /**
     * Constructs a {@link AsArray}.
     *
     * @param field the output field name
     * @param expectedDim expected number of dimensions
     * @param dataType {@link DataType} to use
     */
    public AsArray(FieldName field, int expectedDim, DataType dataType) {
        this.field = field;
        this.dataType = dataType;
        this.expectedDim = expectedDim;
    }

    /**
     * Constructs a {@link AsArray}.
     *
     * @param field the output field name
     * @param expectedDim expected number of dimensions
     */
    public AsArray(FieldName field, int expectedDim) {
        this(field, expectedDim, DataType.FLOAT32);
    }

    /** {@inheritDoc} */
    @Override
    public TimeSeriesData transform(NDManager manager, TimeSeriesData data, boolean isTrain) {
        Convert.asArray(field, expectedDim, dataType, data);
        return data;
    }
}
