package ai.djl.integration.tests.training.listener;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.integration.util.TestUtils;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.EarlyStoppingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

import java.io.IOException;

public class EarlyStoppingListenerTest {

    private final Optimizer sgd = Optimizer.sgd().setLearningRateTracker(Tracker.fixed(0.1f)).build();

    private Mnist testMnistDataset;
    private Mnist trainMnistDataset;

    @BeforeTest
    public void setUp() throws IOException, TranslateException {
        testMnistDataset = Mnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(32, false)
                .build();
        testMnistDataset.prepare();

        trainMnistDataset = Mnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(32, false)
                .build();
        trainMnistDataset.prepare();
    }

    @Test
    public void testEarlyStoppingStopsOnEpoch2() throws Exception {
        Mlp mlpModel = new Mlp(784, 1, new int[]{256}, Activation::relu);

        try (Model model = Model.newInstance("lin-reg", TestUtils.getEngine())) {
            model.setBlock(mlpModel);

            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                    .optOptimizer(sgd)
                    .addTrainingListeners(TrainingListener.Defaults.logging())
                    .addTrainingListeners(new EarlyStoppingListener()
                            .setEpochPatience(1)
                            .setEarlyStopPctImprovement(50)
                            .setMaxMinutes(60)
                            .setMinEpochs(1)
                    );

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, 784));
                Metrics metrics = new Metrics();
                trainer.setMetrics(metrics);

                try {
                    // Set epoch to 5 as we expect the early stopping to stop after the second epoch
                    EasyTrain.fit(trainer, 5, trainMnistDataset, testMnistDataset);
                } catch (EarlyStoppingListener.EarlyStoppedException e) {
                    Assert.assertEquals(e.getMessage(), "failed to achieve 50.0% improvement 1 times in a row");
                    Assert.assertEquals(e.getStopEpoch(), 2);
                }

                TrainingResult trainingResult = trainer.getTrainingResult();
                Assert.assertEquals(trainingResult.getEpoch(), 2);
            }
        }
    }

    @Test
    public void testEarlyStoppingStopsOnEpoch3AsMinEpochsIs3() throws Exception {
        Mlp mlpModel = new Mlp(784, 1, new int[]{256}, Activation::relu);

        try (Model model = Model.newInstance("lin-reg", TestUtils.getEngine())) {
            model.setBlock(mlpModel);

            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                    .optOptimizer(sgd)
                    .addTrainingListeners(TrainingListener.Defaults.logging())
                    .addTrainingListeners(new EarlyStoppingListener(0, 3, 60, 50, 1));

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, 784));
                Metrics metrics = new Metrics();
                trainer.setMetrics(metrics);

                try {
                    // Set epoch to 5 as we expect the early stopping to stop after the second epoch
                    EasyTrain.fit(trainer, 5, trainMnistDataset, testMnistDataset);
                } catch (EarlyStoppingListener.EarlyStoppedException e) {
                    Assert.assertEquals(e.getMessage(), "failed to achieve 50.0% improvement 1 times in a row");
                    Assert.assertEquals(e.getStopEpoch(), 3);
                }

                TrainingResult trainingResult = trainer.getTrainingResult();
                Assert.assertEquals(trainingResult.getEpoch(), 3);
            }
        }
    }

}