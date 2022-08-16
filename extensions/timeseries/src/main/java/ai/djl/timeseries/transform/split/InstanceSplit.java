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
package ai.djl.timeseries.transform.split;

import ai.djl.ndarray.NDManager;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.transform.InstanceSampler;
import ai.djl.timeseries.transform.TimeSeriesTransform;

import java.util.List;

/**
 * Use the {@link ai.djl.timeseries.transform.InstanceSampler} to split the time series data into
 * past and future part.
 */
public class InstanceSplit implements TimeSeriesTransform {

    private FieldName targetField;
    private FieldName isPadField;
    private FieldName startField;
    private FieldName forecastStartField;
    private InstanceSampler instanceSampler;
    private int pastLength;
    private int futureLength;
    private int leadTime;
    private boolean outputNTC;
    private List<FieldName> timeSeriesFields;
    private float dummyValue;

    /**
     * Constructs a {@link InstanceSplit}.
     *
     * @param targetField target field name
     * @param isPadField is_pad field name
     * @param startField start time field name
     * @param forecastStartField forecast time field name
     * @param instanceSampler Sampler to generate indices for splitting
     * @param pastLength past target data length
     * @param futureLength future target data length
     * @param leadTime lead time
     * @param outputNTC whether to lay out as "NTC"
     * @param timeSeriesFields time series field names to be split
     * @param dummyValue value for padding
     */
    public InstanceSplit(
            FieldName targetField,
            FieldName isPadField,
            FieldName startField,
            FieldName forecastStartField,
            InstanceSampler instanceSampler,
            int pastLength,
            int futureLength,
            int leadTime,
            boolean outputNTC,
            List<FieldName> timeSeriesFields,
            float dummyValue) {
        this.targetField = targetField;
        this.isPadField = isPadField;
        this.startField = startField;
        this.forecastStartField = forecastStartField;
        this.instanceSampler = instanceSampler;
        this.pastLength = pastLength;
        this.futureLength = futureLength;
        this.leadTime = leadTime;
        this.outputNTC = outputNTC;
        this.timeSeriesFields = timeSeriesFields;
        this.dummyValue = dummyValue;
    }

    /**
     * Constructs a {@link InstanceSplit}.
     *
     * @param targetField target field name
     * @param isPadField is_pad field name
     * @param startField start time field name
     * @param forecastStartField forecast time field name
     * @param instanceSampler Sampler to generate indices for splitting
     * @param pastLength past target data length
     * @param futureLength future target data length
     * @param timeSeriesFields time series field names to be split
     * @param dummyValue value for padding
     */
    public InstanceSplit(
            FieldName targetField,
            FieldName isPadField,
            FieldName startField,
            FieldName forecastStartField,
            InstanceSampler instanceSampler,
            int pastLength,
            int futureLength,
            List<FieldName> timeSeriesFields,
            float dummyValue) {
        this(
                targetField,
                isPadField,
                startField,
                forecastStartField,
                instanceSampler,
                pastLength,
                futureLength,
                0,
                true,
                timeSeriesFields,
                dummyValue);
    }

    /** {@inheritDoc}. */
    @Override
    public TimeSeriesData transform(NDManager manager, TimeSeriesData data) {
        return Split.instanceSplit(
                manager,
                targetField,
                isPadField,
                startField,
                forecastStartField,
                instanceSampler,
                pastLength,
                futureLength,
                leadTime,
                outputNTC,
                timeSeriesFields,
                dummyValue,
                data);
    }
}
