import ai.djl.Model;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingListener;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.util.ProgressBar;
import java.io.IOException;
import java.nio.file.Paths;

public class TrainingUtils {

    public static void fit(
            Trainer trainer,
            int numEpoch,
            Dataset trainingDataset,
            Dataset validateDataset,
            String outputDir,
            String modelName)
            throws IOException {
        for (int epoch = 0; epoch < numEpoch; epoch++) {
            for (Batch batch : trainer.iterateDataset(trainingDataset)) {
                trainer.trainBatch(batch);
                trainer.step();
                batch.close();
            }

            if (validateDataset != null) {
                for (Batch batch : trainer.iterateDataset(validateDataset)) {
                    trainer.validateBatch(batch);
                    batch.close();
                }
            }
            // reset training and validation metric at end of epoch
            trainer.resetTrainingMetrics();
            // save model at end of each epoch
            if (outputDir != null) {
                Model model = trainer.getModel();
                model.setProperty("Epoch", String.valueOf(epoch));
                model.save(Paths.get(outputDir), modelName);
            }
        }
    }

    public static TrainingListener getTrainingListener(
            ProgressBar trainingProgressBar, ProgressBar validateProgressBar) {
        return new SimpleTrainingListener(trainingProgressBar, validateProgressBar);
    }

    private static final class SimpleTrainingListener implements TrainingListener {

        private ProgressBar trainingProgressBar;
        private ProgressBar validateProgressBar;
        private int trainingProgress;
        private int validateProgress;

        public SimpleTrainingListener(
                ProgressBar trainingProgressBar, ProgressBar validateProgressBar) {
            this.trainingProgressBar = trainingProgressBar;
            this.validateProgressBar = validateProgressBar;
        }

        /** {@inheritDoc} */
        @Override
        public void onTrainingBatch() {
            if (trainingProgressBar != null) {
                trainingProgressBar.update(trainingProgress++);
            }
        }

        /** {@inheritDoc} */
        @Override
        public void onValidationBatch() {
            if (validateProgressBar != null) {
                validateProgressBar.update(validateProgress++);
            }
        }

        /** {@inheritDoc} */
        @Override
        public void onEpoch() {
            trainingProgress = 0;
            validateProgress = 0;
        }
    }
}
