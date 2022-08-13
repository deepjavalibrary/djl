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
import ai.djl.gluonTS.transform.GluonTSTransform;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

import java.time.LocalDateTime;
import java.util.List;
import java.util.function.BiFunction;

/** Add time feature by frequency. */
public class AddTimeFeature implements GluonTSTransform {

    private FieldName startField;
    private FieldName targetField;
    private FieldName outputField;
    private List<BiFunction<NDManager, List<LocalDateTime>, NDArray>> timeFeatures;
    private int predictionLength;
    String freq;

    /**
     * Constructs a {@link AddTimeFeature}.
     *
     * @param startField start field name containing start time
     * @param targetField target field name
     * @param outputField output value field name
     * @param timeFeatures functions to generate time features
     * @param predictionLength prediction length
     * @param freq prediction frequency
     */
    public AddTimeFeature(
            FieldName startField,
            FieldName targetField,
            FieldName outputField,
            List<BiFunction<NDManager, List<LocalDateTime>, NDArray>> timeFeatures,
            int predictionLength,
            String freq) {
        this.startField = startField;
        this.targetField = targetField;
        this.outputField = outputField;
        this.timeFeatures = timeFeatures;
        this.predictionLength = predictionLength;
        this.freq = freq;
    }

    /** {@inheritDoc} */
    @Override
    public GluonTSData transform(NDManager manager, GluonTSData data) {
        return Feature.addTimeFeature(
                manager,
                startField,
                targetField,
                outputField,
                timeFeatures,
                predictionLength,
                freq,
                data);
    }
}
