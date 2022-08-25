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

/** Add age feature through the prediction length. */
public class AddAgeFeature implements TimeSeriesTransform {

    private FieldName targetField;
    private FieldName outputField;
    private int predictionLength;
    private boolean logScale;

    /**
     * Constructs a {@link AddAgeFeature}.
     *
     * @param targetField target field name
     * @param outputField output field name
     * @param predictionLength time series prediction length
     */
    public AddAgeFeature(FieldName targetField, FieldName outputField, int predictionLength) {
        this(targetField, outputField, predictionLength, true);
    }

    /**
     * Constructs a {@link AddAgeFeature}.
     *
     * @param targetField target field name
     * @param outputField output field name
     * @param predictionLength time series prediction length
     * @param logScale whether to use log data
     */
    public AddAgeFeature(
            FieldName targetField, FieldName outputField, int predictionLength, boolean logScale) {
        this.targetField = targetField;
        this.outputField = outputField;
        this.predictionLength = predictionLength;
        this.logScale = logScale;
    }

    /** {@inheritDoc} */
    @Override
    public TimeSeriesData transform(NDManager manager, TimeSeriesData data) {
        return Feature.addAgeFeature(
                manager, targetField, outputField, predictionLength, logScale, data);
    }
}
