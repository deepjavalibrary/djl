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

import ai.djl.ndarray.NDManager;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.transform.TimeSeriesTransform;

/** Add observed value for the target data. */
public class AddObservedValuesIndicator implements TimeSeriesTransform {

    private FieldName targetField;
    private FieldName outputField;

    /**
     * Constructs a {@link AddObservedValuesIndicator}.
     *
     * @param targetField target field name to be observed
     * @param outputField output field name
     */
    public AddObservedValuesIndicator(FieldName targetField, FieldName outputField) {
        this.targetField = targetField;
        this.outputField = outputField;
    }

    /** {@inheritDoc} */
    @Override
    public TimeSeriesData transform(NDManager manager, TimeSeriesData data, boolean isTrain) {
        Feature.addObservedValuesIndicator(manager, targetField, outputField, data);
        return data;
    }
}
