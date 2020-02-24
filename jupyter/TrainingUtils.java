import ai.djl.Model;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
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
            // reset training and validation evaluators at end of epoch
            trainer.endEpoch();
            // save model at end of each epoch
            if (outputDir != null) {
                Model model = trainer.getModel();
                model.setProperty("Epoch", String.valueOf(epoch));
                model.save(Paths.get(outputDir), modelName);
            }
        }
    }
}
