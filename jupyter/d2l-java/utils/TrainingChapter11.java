import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.api.LinePlot;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import java.io.IOException;
import ai.djl.translate.TranslateException;
import ai.djl.basicdataset.AirfoilRandomAccess;

import java.util.ArrayList;
import java.util.Map;

public class TrainingChapter11 {
    /* Ch11 Optimization */
    public static float[] arrayListToFloat(ArrayList<Double> arrayList) {
        float[] ret = new float[arrayList.size()];

        for (int i = 0; i < arrayList.size(); i++) {
            ret[i] = arrayList.get(i).floatValue();
        }
        return ret;
    }

    @FunctionalInterface
    public static interface TrainerConsumer {
        void train(NDList params, NDList states, Map<String, Float> hyperparams);

    }

    public static class LossTime {
        public float[] loss;
        public float[] time;

        public LossTime(float[] loss, float[] time) {
            this.loss = loss;
            this.time = time;
        }
    }

    /**
     * Gets the airfoil dataset 
     */
    public AirfoilRandomAccess getDataCh11(int batchSize, int n) 
            throws IOException, TranslateException {
        // Load data
        AirfoilRandomAccess airfoil =
                AirfoilRandomAccess.builder()
                        .optUsage(Dataset.Usage.TRAIN)
                        .setSampling(batchSize, true)
                        .build();
        // Select Features
        airfoil.addAllFeatures();
        // Prepare Data
        airfoil.prepare();
        // Select first n cases
        airfoil.selectFirstN(n);
        // Remove the mean and rescale variance to 1 for all features
        airfoil.whitenAll();
        return airfoil;
    }
    
    /**
     *  Evaluate the loss of a model on the given dataset
     */
    public static float evaluateLoss(Iterable<Batch> dataIterator, NDArray w, NDArray b) {
        Accumulator metric = new Accumulator(2);  // sumLoss, numExamples

        for (Batch batch : dataIterator) {
            NDArray X = batch.getData().head();
            NDArray y = batch.getLabels().head();
            NDArray yHat = Training.linreg(X, w, b);
            float lossSum = Training.squaredLoss(yHat, y).sum().getFloat();

            metric.add(new float[]{lossSum, (float) y.size()});
            batch.close();
        }
        return metric.get(0) / metric.get(1);
    }

    public static void plotLossEpoch(float[] loss, float[] epoch) {
        Table data = Table.create("data")
                .addColumns(
                        DoubleColumn.create("epoch", Functions.floatToDoubleArray(epoch)),
                        DoubleColumn.create("loss", Functions.floatToDoubleArray(loss))
                );
        display(LinePlot.create("loss vs. epoch", data, "epoch", "loss"));
    }

    public static LossTime trainCh11(TrainerConsumer trainer, 
                                     NDList states, Map<String, 
                                     Float> hyperparams,
                                     AirfoilRandomAccess dataset,
                                     int featureDim, int numEpochs) {
        NDManager manager = NDManager.newBaseManager();
        NDArray w = manager.randomNormal(0, 0.01f, new Shape(featureDim, 1), DataType.FLOAT32);
        NDArray b = manager.zeros(new Shape(1));

        w.attachGradient();
        b.attachGradient();

        NDList params = new NDList(w, b);
        int n = 0;
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        float lastLoss = -1;
        ArrayList<Double> loss = new ArrayList<>();
        ArrayList<Double> epoch = new ArrayList<>();

        for (int i = 0; i < numEpochs; i++) {
            for (Batch batch : dataset.getData(manager)) {
                int len = (int) dataset.size() / batch.getSize();  // number of batches
                NDArray X = batch.getData().head();
                NDArray y = batch.getLabels().head();

                NDArray l;
                try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                    NDArray yHat = Training.linreg(X, params.get(0), params.get(1));
                    l = Training.squaredLoss(yHat, y).mean();
                    gc.backward(l);
                }

                trainer.train(params, states, hyperparams);
                n += X.getShape().get(0);

                if (n % 200 == 0) {
                    stopWatch.stop();
                    lastLoss = evaluateLoss(dataset.getData(manager), params.get(0), params.get(1));
                    loss.add((double) lastLoss);
                    double lastEpoch = 1.0 * n / X.getShape().get(0) / len;
                    epoch.add(lastEpoch);
                    stopWatch.start();
                }

                batch.close();
            }
        }
        plotLossEpoch(arrayListToFloat(loss), arrayListToFloat(epoch));
        System.out.printf("loss: %.3f, %.3f sec/epoch\n", lastLoss, stopWatch.avg());
        return new LossTime(arrayListToFloat(loss), arrayListToFloat(stopWatch.cumsum()));
    }

    public static void trainConciseCh11(Optimizer sgd, AirfoilRandomAccess dataset,
                                        int numEpochs) {
        // Initialization
        NDManager manager = NDManager.newBaseManager();

        SequentialBlock net = new SequentialBlock();
        Linear linear = Linear.builder().setUnits(1).build();
        net.add(linear);
        net.setInitializer(new NormalInitializer());

        Model model = Model.newInstance("concise implementation");
        model.setBlock(net);

        Loss loss = Loss.l2Loss();

        DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                .optOptimizer(sgd)
                .addEvaluator(new Accuracy()) // Model Accuracy
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        Trainer trainer = model.newTrainer(config);

        int n = 0;
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        trainer.initialize(new Shape(10, 5));

        Metrics metrics = new Metrics();
        trainer.setMetrics(metrics);

        float lastLoss = -1;

        ArrayList<Double> lossArray = new ArrayList<>();
        ArrayList<Double> epochArray = new ArrayList<>();

        for (Batch batch : trainer.iterateDataset(dataset)) {
            int len = (int) dataset.size() / batch.getSize();  // number of batches

            NDArray X = batch.getData().head();
            EasyTrain.trainBatch(trainer, batch);
            trainer.step();

            n += X.getShape().get(0);

            if (n % 200 == 0) {
                stopWatch.stop();
                lastLoss = evaluateLoss(dataset.getData(manager), linear.getParameters().get(0).getValue().getArray()
                                .reshape(new Shape(dataset.getFeatureArraySize(), 1)),
                        linear.getParameters().get(1).getValue().getArray());

                lossArray.add((double) lastLoss);
                double lastEpoch = 1.0 * n / X.getShape().get(0) / len;
                epochArray.add(lastEpoch);
                stopWatch.start();
            }
            batch.close();
        }
        plotLossEpoch(arrayListToFloat(lossArray), arrayListToFloat(epochArray));

        System.out.printf("loss: %.3f, %.3f sec/epoch\n", lastLoss, stopWatch.avg());
    }
    /* End Ch11 Optimization */
}
