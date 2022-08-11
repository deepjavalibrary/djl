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

package ai.djl.gluonTS.transform;

import ai.djl.gluonTS.GluonTSData;
import ai.djl.gluonTS.dataset.FieldName;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;

public class Split {

    public Split() {}

    /**
     * Selects training instances, by slicing the target and other time series like arrays at random
     * points in training mode or at the last time point in prediction mode. Assumption is that all
     * time like arrays start at the same time point.
     *
     * <p>The target and each time_series_field is removed and instead two corresponding fields with
     * prefix `past_` and `future_` are included. E.g.
     *
     * <p>If the target array is one-dimensional, the resulting instance has shape (len_target). In
     * the multi-dimensional case, the instance has shape (dim, len_target).
     *
     * <p>target -> past_target and future_target
     *
     * <p>The transformation also adds a field 'past_is_pad' that indicates whether values where
     * padded or not. Convention: time axis is always the last axis.
     *
     * @param manager the default {@link NDManager}
     * @param targetField {@link FieldName} containing the target
     * @param isPadField output {@link FieldName} indicating whether padding happened
     * @param startField field containing the start date of the time series
     * @param forecastStartField output field that will contain the time point where the forecast
     *     starts
     * @param instanceSampler {@link InstanceSampler} that provides sampling indices given a
     *     time-series
     * @param pastLength length of the target seen before making prediction
     * @param futureLength length of the target that must be predicted
     * @param leadTime gap between the past and future windows (default: 0)
     * @param outputNTC whether to have time series output in (time, dimension) or in (dimension,
     *     time) layout (default: True)
     * @param timeSeriesFields fields that contains time-series, they are split in the same interval
     *     as the target (default: None)
     * @param dummyValue Value to use for padding. (default: 0.0)
     * @param data the {@link GluonTSData} to operate on
     * @return the result {@link GluonTSData}
     */
    public static GluonTSData instanceSplit(
            NDManager manager,
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
            float dummyValue,
            GluonTSData data) {

        List<FieldName> sliceCols = new ArrayList<>(timeSeriesFields);
        sliceCols.add(targetField);
        NDArray target = data.get(targetField);

        List<Integer> sampledIndices = instanceSampler.call(target);

        // TODO: add yield method
        for (int i : sampledIndices.subList(0, 1)) {
            int padLength = Math.max(pastLength - i, 0);
            for (FieldName tsField : sliceCols) {
                NDArray pastPiece;
                NDArray tsData = data.get(tsField);
                if (i > pastLength) {
                    pastPiece = tsData.get("..., {}:{}", i - pastLength, i);
                } else if (i < pastLength) {
                    Shape shape = tsData.getShape();
                    int dims = shape.dimension();
                    shape = shape.slice(0, dims - 1).add(padLength);
                    NDArray padBlock = manager.full(shape, dummyValue, tsData.getDataType());

                    pastPiece = padBlock.concat(tsData.get("..., :{}", i), -1);
                } else {
                    pastPiece = tsData.get("..., :{}", i);
                }
                data.setField(_past(tsField), pastPiece);
                data.setField(
                        _future(tsField),
                        tsData.get("..., {}:{}", i + leadTime, i + leadTime + futureLength));
                data.remove(tsField);
            }

            NDArray padIndicator = manager.zeros(new Shape(pastLength), target.getDataType());
            if (padLength > 0) {
                padIndicator.set(new NDIndex(":{}", padLength), 1);
            }

            if (outputNTC) {
                for (FieldName tsField : sliceCols) {
                    NDArray past = data.get(_past(tsField));
                    data.setField(_past(tsField), past.transpose());
                    NDArray future = data.get(_future(tsField));
                    data.setField(_future(tsField), future.transpose());
                }
            }

            data.setField(_past(isPadField), padIndicator);

            // only for freq "D" now
            data.setForeCastStartTime(data.getStartTime().plusDays(i + leadTime));
        }
        return data;
    }

    /**
     * Selects training instances, by slicing the target and other time series like arrays at random
     * points in training mode or at the last time point in prediction mode. Assumption is that all
     * time like arrays start at the same time point.
     *
     * <p>The target and each time_series_field is removed and instead two corresponding fields with
     * prefix `past_` and `future_` are included. E.g.
     *
     * <p>If the target array is one-dimensional, the resulting instance has shape (len_target). In
     * the multi-dimensional case, the instance has shape (dim, len_target).
     *
     * <p>target -> past_target and future_target
     *
     * <p>The transformation also adds a field 'past_is_pad' that indicates whether values where
     * padded or not. Convention: time axis is always the last axis.
     *
     * @param manager the default {@link NDManager}
     * @param targetField {@link FieldName} containing the target
     * @param isPadField output {@link FieldName} indicating whether padding happened
     * @param startField field containing the start date of the time series
     * @param forecastStartField output field that will contain the time point where the forecast
     *     starts
     * @param instanceSampler {@link InstanceSampler} that provides sampling indices given a
     *     time-series
     * @param pastLength length of the target seen before making prediction
     * @param futureLength length of the target that must be predicted (time, dimension) or in
     *     (dimension, time) layout (default: True)
     * @param timeSeriesFields fields that contains time-series, they are split in the same interval
     *     as the target (default: None)
     * @param dummyValue Value to use for padding. (default: 0.0)
     * @param data the {@link GluonTSData} to operate on
     * @return the result {@link GluonTSData}
     */
    public static GluonTSData instanceSplit(
            NDManager manager,
            FieldName targetField,
            FieldName isPadField,
            FieldName startField,
            FieldName forecastStartField,
            InstanceSampler instanceSampler,
            int pastLength,
            int futureLength,
            List<FieldName> timeSeriesFields,
            float dummyValue,
            GluonTSData data) {
        return instanceSplit(
                manager,
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
                dummyValue,
                data);
    }

    private static String _past(FieldName name) {
        return String.format("PAST_%s", name.name());
    }

    private static String _future(FieldName name) {
        return String.format("FUTURE_%s", name.name());
    }
}
