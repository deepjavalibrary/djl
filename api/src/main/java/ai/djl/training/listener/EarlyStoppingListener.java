/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;

import java.time.Duration;

/**
 * Listener that allows the training to be stopped early if the validation loss is not improving, or
 * if time has expired. <br>
 *
 * <p>Usage: Add this listener to the training config, and add it as the last one.
 *
 * <pre>
 *  new DefaultTrainingConfig(...)
 *        .addTrainingListeners(EarlyStoppingListener.builder()
 *                .setEpochPatience(1)
 *                .setEarlyStopPctImprovement(1)
 *                .setMaxDuration(Duration.ofMinutes(42))
 *                .setMinEpochs(1)
 *                .build()
 *        );
 * </pre>
 *
 * <p>Then surround the fit with a try catch that catches the {@link
 * EarlyStoppingListener.EarlyStoppedException}. <br>
 * Example:
 *
 * <pre>
 * try {
 *   EasyTrain.fit(trainer, 5, trainDataset, testDataset);
 * } catch (EarlyStoppingListener.EarlyStoppedException e) {
 *   // handle early stopping
 *   log.info("Stopped early at epoch {} because: {}", e.getEpoch(), e.getMessage());
 * }
 * </pre>
 *
 * <br>
 * Note: Ensure that Metrics are set on the trainer.
 */
public final class EarlyStoppingListener implements TrainingListener {
    private final double objectiveSuccess;

    private final int minEpochs;
    private final long maxMillis;
    private final double earlyStopPctImprovement;
    private final int epochPatience;

    private long startTimeMills;
    private double prevLoss;
    private int numberOfEpochsWithoutImprovements;

    private final String monitoredMetric;

    private EarlyStoppingListener(
            double objectiveSuccess,
            int minEpochs,
            long maxMillis,
            double earlyStopPctImprovement,
            int earlyStopPatience,
            String monitoredMetric) {
        this.objectiveSuccess = objectiveSuccess;
        this.minEpochs = minEpochs;
        this.maxMillis = maxMillis;
        this.earlyStopPctImprovement = earlyStopPctImprovement;
        this.epochPatience = earlyStopPatience;
        this.monitoredMetric = monitoredMetric;
    }

    /** {@inheritDoc} */
    @Override
    public void onEpoch(Trainer trainer) {
        int currentEpoch = trainer.getTrainingResult().getEpoch();
        // stopping criteria
        final double loss = getLoss(trainer.getTrainingResult());
        if (currentEpoch >= minEpochs) {
            if (loss < objectiveSuccess) {
                throw new EarlyStoppedException(
                        currentEpoch,
                        String.format(
                                "validation loss %s < objectiveSuccess %s",
                                loss, objectiveSuccess));
            }
            long elapsedMillis = System.currentTimeMillis() - startTimeMills;
            if (elapsedMillis >= maxMillis) {
                throw new EarlyStoppedException(
                        currentEpoch,
                        String.format("%s ms elapsed >= %s maxMillis", elapsedMillis, maxMillis));
            }
            // consider early stopping?
            if (Double.isFinite(prevLoss)) {
                double goalImprovement = prevLoss * (100 - earlyStopPctImprovement) / 100.0;
                boolean improved = loss <= goalImprovement; // false if any NANs
                if (improved) {
                    numberOfEpochsWithoutImprovements = 0;
                } else {
                    numberOfEpochsWithoutImprovements++;
                    if (numberOfEpochsWithoutImprovements >= epochPatience) {
                        throw new EarlyStoppedException(
                                currentEpoch,
                                String.format(
                                        "failed to achieve %s%% improvement %s times in a row",
                                        earlyStopPctImprovement, epochPatience));
                    }
                }
            }
        }
        if (Double.isFinite(loss)) {
            prevLoss = loss;
        }
    }

    private static double getMetric(TrainingResult trainingResult) {
        if ("validateLoss".equals(monitoredMetric)) {
            Float vLoss = trainingResult.getValidateLoss();
            return vLoss != null ? vLoss : Double.NaN;
        } else if ("trainLoss".equals(monitoredMetric)) {
            Float tLoss = trainingResult.getTrainLoss();
            return tLoss != null ? tLoss : Double.NaN;
        } else {
            Float val = trainingResult.getEvaluations().get(monitoredMetric);
            return val != null ? val : Double.NaN;
        }
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {
        // do nothing
    }

    /** {@inheritDoc} */
    @Override
    public void onValidationBatch(Trainer trainer, BatchData batchData) {
        // do nothing
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBegin(Trainer trainer) {
        this.startTimeMills = System.currentTimeMillis();
        this.prevLoss = Double.NaN;
        this.numberOfEpochsWithoutImprovements = 0;
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingEnd(Trainer trainer) {
        // do nothing
    }

    /**
     * Creates a builder to build a {@link EarlyStoppingListener}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A builder for a {@link EarlyStoppingListener}. */
    public static final class Builder {
        private final double objectiveSuccess;
        private int minEpochs;
        private long maxMillis;
        private double earlyStopPctImprovement;
        private int epochPatience;
        private String monitoredMetric;

        /** Constructs a {@link Builder} with default values. */
        public Builder() {
            this.objectiveSuccess = 0;
            this.minEpochs = 0;
            this.maxMillis = Long.MAX_VALUE;
            this.earlyStopPctImprovement = 0;
            this.epochPatience = 0;
            this.monitoredMetric = "validateLoss";
        }

        /**
         * Set the minimum # epochs, defaults to 0.
         *
         * @param minEpochs the minimum # epochs
         * @return this builder
         */
        public Builder optMinEpochs(int minEpochs) {
            this.minEpochs = minEpochs;
            return this;
        }

        /**
         * Set the maximum duration a training run should take, defaults to Long.MAX_VALUE in ms.
         *
         * @param duration the maximum duration a training run should take
         * @return this builder
         */
        public Builder optMaxDuration(Duration duration) {
            this.maxMillis = duration.toMillis();
            return this;
        }

        /**
         * Set the maximum # milliseconds a training run should take, defaults to Long.MAX_VALUE.
         *
         * @param maxMillis the maximum # milliseconds a training run should take
         * @return this builder
         */
        public Builder optMaxMillis(int maxMillis) {
            this.maxMillis = maxMillis;
            return this;
        }

        /**
         * Consider early stopping if not x% improvement, defaults to 0.
         *
         * @param earlyStopPctImprovement the percentage improvement to consider early stopping,
         *     must be between 0 and 100.
         * @return this builder
         */
        public Builder optEarlyStopPctImprovement(double earlyStopPctImprovement) {
            this.earlyStopPctImprovement = earlyStopPctImprovement;
            return this;
        }

        /**
         * Stop if insufficient improvement for x epochs in a row, defaults to 0.
         *
         * @param epochPatience the number of epochs without improvement to consider stopping, must
         *     be greater than 0.
         * @return this builder
         */
        public Builder optEpochPatience(int epochPatience) {
            this.epochPatience = epochPatience;
            return this;
        }

        /**
         * Builds a {@link EarlyStoppingListener} with the specified values.
         *
         * @return a new {@link EarlyStoppingListener}
         */
        public EarlyStoppingListener build() {
            return new EarlyStoppingListener(
                    objectiveSuccess, minEpochs, maxMillis, earlyStopPctImprovement, epochPatience, monitoredMetric);
        }
    }

    /**
     * Thrown when training is stopped early, the message will contain the reason why it is stopped
     * early.
     */
    public static class EarlyStoppedException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        private final int stopEpoch;

        /**
         * Constructs an {@link EarlyStoppedException} with the specified message and epoch.
         *
         * @param stopEpoch the epoch at which training was stopped early
         * @param message the message/reason why training was stopped early
         */
        public EarlyStoppedException(int stopEpoch, String message) {
            super(message);
            this.stopEpoch = stopEpoch;
        }

        /**
         * Gets the epoch at which training was stopped early.
         *
         * @return the epoch at which training was stopped early.
         */
        public int getStopEpoch() {
            return stopEpoch;
        }
    }
}
