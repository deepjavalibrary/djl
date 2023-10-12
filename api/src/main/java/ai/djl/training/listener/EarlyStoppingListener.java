package ai.djl.training.listener;

import ai.djl.training.Trainer;

/**
 * Listener that allows the training to be stopped early if the validation loss is not improving, or if time has expired.
 * <br/>
 * Usage:
 * Add this listener to the training config, and add it as the last one.
 * <pre>
 *  new DefaultTrainingConfig(...)
 *        .addTrainingListeners(new EarlyStoppingListener()
 *                .setEpochPatience(1)
 *                .setEarlyStopPctImprovement(1)
 *                .setMaxMinutes(60)
 *                .setMinEpochs(1)
 *        );
 * </pre>
 * Then surround the fit with a try catch that catches the {@link EarlyStoppingListener.EarlyStoppedException}.
 * <br/>
 * Example:
 * <pre>
 * try {
 *   EasyTrain.fit(trainer, 5, trainDataset, testDataset);
 * } catch (EarlyStoppingListener.EarlyStoppedException e) {
 *   // handle early stopping
 *   log.info("Stopped early at epoch {} because: {}", e.getEpoch(), e.getMessage());
 * }
 * </pre>
 * <br/>
 * Note: Ensure that Metrics are set on the trainer.
 */
public class EarlyStoppingListener implements TrainingListener {
    private final double objectiveSuccess;

    /**
     * after minimum # epochs, consider stopping if:
     */
    private int minEpochs;
    /**
     * too much time elapsed
     */
    private int maxMinutes;
    /**
     * consider early stopping if not x% improvement
     */
    private double earlyStopPctImprovement;
    /**
     * stop if insufficient improvement for x epochs in a row
     */
    private int epochPatience;

    private long startTimeMills;
    private double prevLoss;
    private int numberOfEpochsWithoutImprovements;

    public EarlyStoppingListener() {
        this.objectiveSuccess = 0;
        this.minEpochs = 0;
        this.maxMinutes = Integer.MAX_VALUE;
        this.earlyStopPctImprovement = 0;
        this.epochPatience = 0;
    }

    public EarlyStoppingListener(double objectiveSuccess, int minEpochs, int maxMinutes, double earlyStopPctImprovement, int earlyStopPatience) {
        this.objectiveSuccess = objectiveSuccess;
        this.minEpochs = minEpochs;
        this.maxMinutes = maxMinutes;
        this.earlyStopPctImprovement = earlyStopPctImprovement;
        this.epochPatience = earlyStopPatience;
    }

    @Override
    public void onEpoch(Trainer trainer) {
        int currentEpoch = trainer.getTrainingResult().getEpoch();
        // stopping criteria
        final double loss = getLoss(trainer);
        if (loss < objectiveSuccess) {
            throw new EarlyStoppedException(currentEpoch, String.format("validation loss %s < objectiveSuccess %s", loss, objectiveSuccess));
        }
        if (currentEpoch >= minEpochs) {
            double elapsedMinutes = (System.currentTimeMillis() - startTimeMills) / 60_000.0;
            if (elapsedMinutes >= maxMinutes) {
                throw new EarlyStoppedException(currentEpoch, String.format("Early stopping training: %.1f minutes elapsed >= %s maxMinutes", elapsedMinutes, maxMinutes));
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
                        throw new EarlyStoppedException(currentEpoch, String.format("failed to achieve %s%% improvement %s times in a row",
                                earlyStopPctImprovement, epochPatience));
                    }
                }
            }
        }
        if (Double.isFinite(loss)) {
            prevLoss = loss;
        }
    }

    private static double getLoss(Trainer trainer) {
        Float vLoss = trainer.getTrainingResult().getValidateLoss();
        if (vLoss != null) {
            return vLoss;
        }
        Float tLoss = trainer.getTrainingResult().getTrainLoss();
        if (tLoss == null) {
            return Double.NaN;
        }
        return tLoss;
    }

    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {
        // do nothing
    }

    @Override
    public void onValidationBatch(Trainer trainer, BatchData batchData) {
        // do nothing
    }

    @Override
    public void onTrainingBegin(Trainer trainer) {
        this.startTimeMills = System.currentTimeMillis();
        this.prevLoss = Double.NaN;
        this.numberOfEpochsWithoutImprovements = 0;
    }

    @Override
    public void onTrainingEnd(Trainer trainer) {
        // do nothing
    }

    public EarlyStoppingListener setMinEpochs(int minEpochs) {
        this.minEpochs = minEpochs;
        return this;
    }

    public EarlyStoppingListener setMaxMinutes(int maxMinutes) {
        this.maxMinutes = maxMinutes;
        return this;
    }

    public EarlyStoppingListener setEarlyStopPctImprovement(double earlyStopPctImprovement) {
        this.earlyStopPctImprovement = earlyStopPctImprovement;
        return this;
    }

    public EarlyStoppingListener setEpochPatience(int epochPatience) {
        this.epochPatience = epochPatience;
        return this;
    }

    /**
     * Thrown when training is stopped early, the message will contain the reason why it is stopped early.
     */
    public static class EarlyStoppedException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        private final int stopEpoch;
        public EarlyStoppedException(int stopEpoch, String message) {
            super(message);
            this.stopEpoch = stopEpoch;
        }

        public int getStopEpoch() {
            return stopEpoch;
        }
    }
}
