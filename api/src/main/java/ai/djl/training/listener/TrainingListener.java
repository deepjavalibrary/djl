/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.training.listener;

import ai.djl.Device;
import ai.djl.ndarray.NDList;
import ai.djl.training.Trainer;
import java.util.Map;

/**
 * {@code TrainingListener} offers an interface that allows performing some actions when certain
 * events have occurred in the {@link Trainer}.
 *
 * <p>The methods {@link #onEpoch(Trainer) onEpoch}, {@link #onTrainingBatch(Trainer, BatchData)
 * onTrainingBatch}, {@link #onValidationBatch(Trainer, BatchData) onValidationBatch} are called
 * during training. Adding an implementation of the listener to the {@link Trainer} allows
 * performing any desired actions at those junctures. These could be used for collection metrics, or
 * logging, or any other purpose.
 */
public interface TrainingListener {

    /**
     * Listens to the end of an epoch during training.
     *
     * @param trainer the trainer the listener is attached to
     */
    void onEpoch(Trainer trainer);

    /**
     * Listens to the end of training one batch of data during training.
     *
     * @param trainer the trainer the listener is attached to
     * @param batchData the data from the batch
     */
    void onTrainingBatch(Trainer trainer, BatchData batchData);

    /**
     * Listens to the end of validating one batch of data during validation.
     *
     * @param trainer the trainer the listener is attached to
     * @param batchData the data from the batch
     */
    void onValidationBatch(Trainer trainer, BatchData batchData);

    /**
     * Listens to the beginning of training.
     *
     * @param trainer the trainer the listener is attached to
     */
    void onTrainingBegin(Trainer trainer);

    /**
     * Listens to the end of training.
     *
     * @param trainer the trainer the listener is attached to
     */
    void onTrainingEnd(Trainer trainer);

    /** Contains default {@link TrainingListener} sets. */
    interface Defaults {

        /**
         * A default {@link TrainingListener} set including batch output logging.
         *
         * @param batchSize the size of training batches
         * @param trainDataSize the total number of elements in the training dataset
         * @param validateDataSize the total number of elements in the validation dataset
         * @param outputDir the output directory to store created log files
         * @return the new set of listeners
         */
        static TrainingListener[] logging(
                int batchSize, int trainDataSize, int validateDataSize, String outputDir) {
            return new TrainingListener[] {
                new EpochTrainingListener(),
                new MemoryTrainingListener(outputDir),
                new EvaluatorTrainingListener(),
                new DivergenceCheckTrainingListener(),
                new LoggingTrainingListener(batchSize, trainDataSize, validateDataSize),
                new TimeMeasureTrainingListener(outputDir)
            };
        }
    }

    /** A class to pass data from the batch into the training listeners. */
    class BatchData {

        private Map<Device, NDList> labels;
        private Map<Device, NDList> predictions;

        /**
         * Constructs a new {@link BatchData}.
         *
         * @param labels the labels for each device
         * @param predictions the predictions for each device
         */
        public BatchData(Map<Device, NDList> labels, Map<Device, NDList> predictions) {
            this.labels = labels;
            this.predictions = predictions;
        }

        /**
         * Returns the labels for each device.
         *
         * @return the labels for each device
         */
        public Map<Device, NDList> getLabels() {
            return labels;
        }

        /**
         * Returns the predictions for each device.
         *
         * @return the predictions for each device
         */
        public Map<Device, NDList> getPredictions() {
            return predictions;
        }
    }
}
