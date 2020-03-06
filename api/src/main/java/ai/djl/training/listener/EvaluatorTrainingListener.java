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
import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Evaluator;

/**
 * {@link TrainingListener} that records evaluator results.
 *
 * <p>Results are recorded for the following stages:
 *
 * <ul>
 *   <li>{@link #TRAIN_EPOCH} - This accumulates for the whole epoch and is recorded to a metric at
 *       the end of the epoch
 *   <li>{@link #TRAIN_PROGRESS} - This accumulates for {@link #progressUpdateFrequency} batches and
 *       is recorded to a metric at the end
 *   <li>{@link #TRAIN_ALL} - This does not accumulates and records every training batch to a metric
 *   <li>{@link #VALIDATE_EPOCH} - This accumulates for the whole validation epoch and is recorded
 *       to a metric at the end of the epoch
 * </ul>
 *
 * <p>The training and validation evaluators are saved as metrics with names that can be found using
 * {@link EvaluatorTrainingListener#metricName(Evaluator, String)}. The validation evaluators are
 * also saved as model properties with the evaluator name.
 */
public class EvaluatorTrainingListener implements TrainingListener {

    public static final String TRAIN_EPOCH = "train/epoch";
    public static final String TRAIN_PROGRESS = "train/progress";
    public static final String TRAIN_ALL = "train/all";
    public static final String VALIDATE_EPOCH = "validate/epoch";

    private int progressUpdateFrequency;

    private int progressCounter;

    /**
     * Constructs an {@link EvaluatorTrainingListener} that updates the training progress the
     * default frequency.
     *
     * <p>Current default frequency is every 5 batches.
     */
    public EvaluatorTrainingListener() {
        this(5);
    }

    /**
     * Constructs an {@link EvaluatorTrainingListener} that updates the training progress the given
     * frequency.
     *
     * @param progressUpdateFrequency the number of batches to accumulate an evaluator before it is
     *     stable enough to output
     */
    public EvaluatorTrainingListener(int progressUpdateFrequency) {
        this.progressUpdateFrequency = progressUpdateFrequency;
        progressCounter = 0;
    }

    /** {@inheritDoc} */
    @Override
    public void onEpoch(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        if (metrics != null) {
            for (Evaluator evaluator : trainer.getEvaluators()) {
                metrics.addMetric(
                        metricName(evaluator, TRAIN_EPOCH), evaluator.getAccumulator(TRAIN_EPOCH));
            }
        }
        for (Evaluator evaluator : trainer.getEvaluators()) {
            evaluator.resetAccumulator(TRAIN_EPOCH);
            evaluator.resetAccumulator(TRAIN_PROGRESS);
            evaluator.resetAccumulator(TRAIN_ALL);
            evaluator.resetAccumulator(VALIDATE_EPOCH);
        }
        progressCounter = 0;
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {
        for (Evaluator evaluator : trainer.getEvaluators()) {
            evaluator.resetAccumulator(TRAIN_ALL);
        }

        updateEvaluators(trainer, batchData, new String[] {TRAIN_EPOCH, TRAIN_PROGRESS, TRAIN_ALL});
        Metrics metrics = trainer.getMetrics();
        if (metrics != null) {
            for (Evaluator evaluator : trainer.getEvaluators()) {
                metrics.addMetric(
                        metricName(evaluator, TRAIN_ALL), evaluator.getAccumulator(TRAIN_ALL));
            }

            progressCounter++;
            if (progressCounter == progressUpdateFrequency) {
                for (Evaluator evaluator : trainer.getEvaluators()) {
                    metrics.addMetric(
                            metricName(evaluator, TRAIN_PROGRESS),
                            evaluator.getAccumulator(TRAIN_PROGRESS));
                }
                progressCounter = 0;
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void onValidationBatch(Trainer trainer, BatchData batchData) {
        updateEvaluators(trainer, batchData, new String[] {VALIDATE_EPOCH});
        Metrics metrics = trainer.getMetrics();
        if (metrics != null) {
            for (Evaluator evaluator : trainer.getEvaluators()) {
                metrics.addMetric(
                        metricName(evaluator, VALIDATE_EPOCH),
                        evaluator.getAccumulator(VALIDATE_EPOCH));
            }
        }
    }

    private void updateEvaluators(Trainer trainer, BatchData batchData, String[] accumulators) {
        for (Evaluator evaluator : trainer.getEvaluators()) {
            for (Device device : batchData.getLabels().keySet()) {
                NDList labels = batchData.getLabels().get(device);
                NDList predictions = batchData.getPredictions().get(device);
                for (String accumulator : accumulators) {
                    evaluator.updateAccumulator(accumulator, labels, predictions);
                }
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBegin(Trainer trainer) {
        trainer.getEvaluators()
                .forEach(
                        evaluator -> {
                            evaluator.addAccumulator(TRAIN_EPOCH);
                            evaluator.addAccumulator(TRAIN_PROGRESS);
                            evaluator.addAccumulator(TRAIN_ALL);
                            evaluator.addAccumulator(VALIDATE_EPOCH);
                        });
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingEnd(Trainer trainer) {
        Model model = trainer.getModel();
        Metrics metrics = trainer.getMetrics();
        if (metrics != null) {
            for (Evaluator evaluator : trainer.getEvaluators()) {
                String metricName = metricName(evaluator, VALIDATE_EPOCH);
                if (metrics.hasMetric(metricName)) {
                    float value = metrics.latestMetric(metricName).getValue().floatValue();
                    model.setProperty(evaluator.getName(), String.format("%.5f", value));
                }
            }
        }
    }

    /**
     * Returns the metric created with the evaluator for the given stage.
     *
     * @param evaluator the evaluator to read the metric from
     * @param stage one of {@link #TRAIN_EPOCH}, {@link #TRAIN_PROGRESS}, or {@link #VALIDATE_EPOCH}
     * @return the metric name to use
     */
    public static String metricName(Evaluator evaluator, String stage) {
        switch (stage) {
            case TRAIN_EPOCH:
                return "train_epoch_" + evaluator.getName();
            case TRAIN_PROGRESS:
                return "train_progress_" + evaluator.getName();
            case TRAIN_ALL:
                return "train_all_" + evaluator.getName();
            case VALIDATE_EPOCH:
                return "validate_epoch_" + evaluator.getName();
            default:
                throw new IllegalArgumentException("Invalid metric stage");
        }
    }
}
