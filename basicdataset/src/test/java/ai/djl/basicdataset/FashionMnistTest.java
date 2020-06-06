package ai.djl.basicdataset;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Blocks;
import ai.djl.repository.Artifact;
import ai.djl.repository.Repository;
import ai.djl.repository.SimpleUrlRepository;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class FashionMnistTest {

    @Test
    public void testFashMnistLocal() throws IOException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optInitializer(Initializer.ONES);

        try (Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            NDManager manager = model.getNDManager();
            Repository repository = Repository.newInstance("test", "src/test/resources/mlrepo/");

            FashionMnist fashionMnist =
                    FashionMnist.builder()
                            .optManager(manager)
                            .optUsage(Dataset.Usage.TEST)
                            .optRepository(repository)
                            .setSampling(32, true)
                            .build();

            fashionMnist.prepare();
            try (Trainer trainer = model.newTrainer(config)) {
                for (Batch batch : trainer.iterateDataset(fashionMnist)) {
                    Assert.assertEquals(batch.getData().size(), 1);
                    Assert.assertEquals(batch.getLabels().size(), 1);
                    batch.close();
                }
            }
        }
    }
}
