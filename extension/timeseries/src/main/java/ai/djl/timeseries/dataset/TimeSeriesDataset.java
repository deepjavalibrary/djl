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

package ai.djl.timeseries.dataset;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.transform.TimeSeriesTransform;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;

import java.util.List;

/** An abstract class for creating time series datasets. */
public abstract class TimeSeriesDataset extends RandomAccessDataset {

    protected List<TimeSeriesTransform> transformation;
    protected int contextLength;

    static final FieldName[] DATASET_FIELD_NAMES = {
        FieldName.TARGET,
        FieldName.FEAT_STATIC_CAT,
        FieldName.FEAT_STATIC_REAL,
        FieldName.FEAT_DYNAMIC_CAT,
        FieldName.FEAT_DYNAMIC_REAL
    };

    protected TimeSeriesDataset(TimeSeriesBuilder<?> builder) {
        super(builder);
        transformation = builder.transformation;
        contextLength = builder.contextLength;
    }

    /**
     * {@code TimeSeriesDataset} override the get function so that it can preprocess the feature
     * data as timeseries package way.
     *
     * <p>{@inheritDoc}
     */
    @Override
    public Record get(NDManager manager, long index) {
        TimeSeriesData data = getTimeSeriesData(manager, index);

        if (transformation.isEmpty()) {
            // For inference with translator
            return new Record(data.toNDList(), new NDList());
        }
        data = apply(manager, data);

        // For both training and prediction
        if (!data.contains("PAST_" + FieldName.TARGET)) {
            throw new IllegalArgumentException(
                    "Transformation must include InstanceSampler to split data into past and future"
                            + " part");
        }

        if (!data.contains("FUTURE_" + FieldName.TARGET)) {
            // Warning: We do not recommend using TimeSeriesDataset directly to generate the
            // inference input, using Translator instead
            // For prediction without translator, we don't need labels and corresponding
            // FUTURE_TARGET.
            return new Record(data.toNDList(), new NDList());
        }

        // For training, we must have the FUTURE_TARGET label to compute Loss.
        NDArray contextTarget = data.get("PAST_" + FieldName.TARGET).get("{}:", -contextLength + 1);
        NDArray futureTarget = data.get("FUTURE_" + FieldName.TARGET);
        NDList label = new NDList(contextTarget.concat(futureTarget, 0));

        return new Record(data.toNDList(), label);
    }

    /**
     * Return the {@link TimeSeriesData} for the given index from the {@code TimeSeriesDataset}.
     *
     * @param manager the manager to create data
     * @param index the index
     * @return the {@link TimeSeriesData}
     */
    public abstract TimeSeriesData getTimeSeriesData(NDManager manager, long index);

    /**
     * Apply to preprocess transformation on {@link TimeSeriesData}.
     *
     * @param manager default {@link NDManager}
     * @param input data the {@link TimeSeriesData} to operate on
     * @return the transformed data
     */
    private TimeSeriesData apply(NDManager manager, TimeSeriesData input) {
        try (NDManager scope = manager.newSubManager()) {
            input.values().forEach(array -> array.tempAttach(scope));
            for (TimeSeriesTransform transform : transformation) {
                input = transform.transform(manager, input, true);
            }
            input.values().forEach(array -> array.attach(manager));
        }
        return input;
    }

    /**
     * Used to build a {@code TimeSeriesDataset}.
     *
     * @param <T> the builder type
     */
    public abstract static class TimeSeriesBuilder<T extends TimeSeriesBuilder<T>>
            extends RandomAccessDataset.BaseBuilder<T> {

        protected List<TimeSeriesTransform> transformation;
        protected int contextLength;

        /**
         * Set the transformation for data preprocess.
         *
         * @param transformation the transformation
         * @return this builder
         */
        public T setTransformation(List<TimeSeriesTransform> transformation) {
            this.transformation = transformation;
            return self();
        }

        /**
         * Set the model prediction context length.
         *
         * @param contextLength the context length
         * @return this builder
         */
        public T setContextLength(int contextLength) {
            this.contextLength = contextLength;
            return self();
        }
    }
}
