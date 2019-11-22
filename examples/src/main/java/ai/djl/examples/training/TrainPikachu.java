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
package ai.djl.examples.training;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.PikachuDetection;
import ai.djl.examples.training.util.AbstractTraining;
import ai.djl.examples.training.util.Arguments;
import ai.djl.examples.training.util.TrainingUtils;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metric;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.ImageVisualization;
import ai.djl.modality.cv.MultiBoxDetection;
import ai.djl.modality.cv.SingleShotDetectionTranslator;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.SingleShotDetectionLoss;
import ai.djl.training.metrics.BoundingBoxError;
import ai.djl.training.metrics.SingleShotDetectionAccuracy;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.zoo.cv.object_detection.ssd.SingleShotDetection;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class TrainPikachu extends AbstractTraining {

    private float trainingClassAccuracy;
    private float trainingBoundingBoxError;
    private float validationClassAccuracy;
    private float validationBoundingBoxError;
    private static final Logger logger = LoggerFactory.getLogger(TrainPikachu.class);

    public static void main(String[] args) {
        new TrainPikachu().runExample(args);
    }

    @Override
    protected void train(Arguments arguments) throws IOException {
        batchSize = arguments.getBatchSize();

        TrainingConfig config = setupTrainingConfig(arguments);
        try (Model model = Model.newInstance()) {
            model.setBlock(getSsdTrainBlock());
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(metrics);
                trainer.setTrainingListener(this);
                Dataset pikachuDetectionTrain = getDataset(Dataset.Usage.TRAIN, arguments);
                Dataset pikachuDetectionTest = getDataset(Dataset.Usage.TEST, arguments);

                Shape inputShape = new Shape(batchSize, 3, 256, 256);
                trainer.initialize(inputShape);
                TrainingUtils.fit(
                        trainer,
                        arguments.getEpoch(),
                        pikachuDetectionTrain,
                        pikachuDetectionTest,
                        arguments.getOutputDir(),
                        "ssd");
            }

            // save model
            model.setProperty("Epoch", String.valueOf(arguments.getEpoch()));
            model.setProperty("Loss", String.format("%.5f", validationLoss));
            model.setProperty("ClassAccuracy", String.format("%.5f", validationClassAccuracy));
            model.setProperty(
                    "BoundingBoxError", String.format("%.5f", validationBoundingBoxError));
            model.save(Paths.get(arguments.getOutputDir()), "ssd");
        }
    }

    public int predict(String outputDir, String imageFile)
            throws IOException, MalformedModelException, TranslateException {
        try (Model model = Model.newInstance()) {
            float detectionThreshold = 0.6f;
            // load parameters back to original training block
            model.setBlock(getSsdTrainBlock());
            model.load(Paths.get(outputDir), "ssd");
            // append prediction logic at end of training block with parameter loaded
            Block ssdTrain = model.getBlock();
            model.setBlock(getSsdPredictBlock(ssdTrain));
            Path imagePath = Paths.get(imageFile);
            Pipeline pipeline = new Pipeline(new ToTensor());
            List<String> classes = new ArrayList<>();
            classes.add("pikachu");
            SingleShotDetectionTranslator translator =
                    new SingleShotDetectionTranslator.Builder()
                            .setPipeline(pipeline)
                            .setClasses(classes)
                            .optThreshold(detectionThreshold)
                            .build();
            try (Predictor<BufferedImage, DetectedObjects> predictor =
                    model.newPredictor(translator)) {
                BufferedImage image = BufferedImageUtils.fromFile(imagePath);
                DetectedObjects detectedObjects = predictor.predict(image);
                ImageVisualization.drawBoundingBoxes(image, detectedObjects);
                Path out = Paths.get(outputDir).resolve("pikachu_output.png");
                ImageIO.write(image, "png", out.toFile());
                // return number of pikachu detected
                return detectedObjects.getNumberOfObjects();
            }
        }
    }

    @Override
    public String getTrainingStatus(Metrics metrics) {
        StringBuilder sb = new StringBuilder();
        List<Metric> list = metrics.getMetric("train_" + loss.getName());
        trainingLoss = list.get(list.size() - 1).getValue().floatValue();

        list = metrics.getMetric("train_classAccuracy");
        trainingClassAccuracy = list.get(list.size() - 1).getValue().floatValue();

        list = metrics.getMetric("train_boundingBoxError");
        trainingBoundingBoxError = list.get(list.size() - 1).getValue().floatValue();
        sb.append(
                String.format(
                        "loss: %2.3ef, classAccuracy: %.4f, bboxError: %2.3e,",
                        trainingLoss, trainingClassAccuracy, trainingBoundingBoxError));

        list = metrics.getMetric("train");
        if (!list.isEmpty()) {
            float batchTime = list.get(list.size() - 1).getValue().longValue() / 1_000_000_000f;
            sb.append(String.format(" speed: %.2f images/sec", (float) batchSize / batchTime));
        }
        return sb.toString();
    }

    @Override
    public void printTrainingStatus(Metrics metrics) {
        List<Metric> list = metrics.getMetric("train_" + loss.getName());
        trainingLoss = list.get(list.size() - 1).getValue().floatValue();

        list = metrics.getMetric("train_classAccuracy");
        trainingClassAccuracy = list.get(list.size() - 1).getValue().floatValue();

        list = metrics.getMetric("train_boundingBoxError");
        trainingBoundingBoxError = list.get(list.size() - 1).getValue().floatValue();

        logger.info(
                "train loss: {}, train class accuracy: {}, train bounding box error: {}",
                trainingLoss,
                trainingClassAccuracy,
                trainingBoundingBoxError);
        list = metrics.getMetric("validate_" + loss.getName());
        if (!list.isEmpty()) {
            validationLoss = list.get(list.size() - 1).getValue().floatValue();
            list = metrics.getMetric("validate_classAccuracy");
            validationClassAccuracy = list.get(list.size() - 1).getValue().floatValue();
            list = metrics.getMetric("validate_boundingBoxError");
            validationBoundingBoxError = list.get(list.size() - 1).getValue().floatValue();
            logger.info(
                    "validate loss: {}, validate class accuracy: {}, validate bounding box error: {}",
                    validationLoss,
                    validationClassAccuracy,
                    validationBoundingBoxError);
        } else {
            logger.info("validation has not been run.");
        }
    }

    private Dataset getDataset(Dataset.Usage usage, Arguments arguments) throws IOException {
        Pipeline pipeline = new Pipeline(new ToTensor());
        PikachuDetection pikachuDetection =
                new PikachuDetection.Builder()
                        .optUsage(usage)
                        .optPipeline(pipeline)
                        .setSampling(batchSize, true)
                        .build();
        pikachuDetection.prepare(new ProgressBar());
        long maxIterations = arguments.getMaxIterations();

        int dataSize = (int) Math.min(pikachuDetection.size() / batchSize, maxIterations);
        if (usage == Dataset.Usage.TRAIN) {
            trainDataSize = dataSize;
        } else if (usage == Dataset.Usage.TEST) {
            validateDataSize = dataSize;
        }
        return pikachuDetection;
    }

    private TrainingConfig setupTrainingConfig(Arguments arguments) {
        Initializer initializer =
                new XavierInitializer(
                        XavierInitializer.RandomType.UNIFORM, XavierInitializer.FactorType.AVG, 2);
        Optimizer optimizer =
                Optimizer.sgd()
                        .setRescaleGrad(1.0f / batchSize)
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.2f))
                        .optWeightDecays(5e-4f)
                        .build();
        loss = new SingleShotDetectionLoss("ssd_loss");
        return new DefaultTrainingConfig(initializer, loss)
                .setOptimizer(optimizer)
                .setBatchSize(batchSize)
                .addTrainingMetric(new SingleShotDetectionAccuracy("classAccuracy"))
                .addTrainingMetric(new BoundingBoxError("boundingBoxError"))
                .setDevices(Device.getDevices(arguments.getMaxGpus()));
    }

    public static Block getSsdTrainBlock() {
        int[] numFilters = {16, 32, 64};
        SequentialBlock baseBlock = new SequentialBlock();
        for (int numFilter : numFilters) {
            baseBlock.add(SingleShotDetection.getDownSamplingBlock(numFilter));
        }

        List<List<Float>> sizes = new ArrayList<>();
        List<List<Float>> ratios = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            ratios.add(Arrays.asList(1f, 2f, 0.5f));
        }
        sizes.add(Arrays.asList(0.2f, 0.272f));
        sizes.add(Arrays.asList(0.37f, 0.447f));
        sizes.add(Arrays.asList(0.54f, 0.619f));
        sizes.add(Arrays.asList(0.71f, 0.79f));
        sizes.add(Arrays.asList(0.88f, 0.961f));

        return new SingleShotDetection.Builder()
                .setNumClasses(1)
                .setNumFeatures(3)
                .optGlobalPool(true)
                .setRatios(ratios)
                .setSizes(sizes)
                .setBaseNetwork(baseBlock)
                .build();
    }

    public static Block getSsdPredictBlock(Block ssdTrain) {
        // add prediction process
        SequentialBlock ssdPredict = new SequentialBlock();
        ssdPredict.add(ssdTrain);
        ssdPredict.add(
                new LambdaBlock(
                        output -> {
                            NDArray anchors = output.get(0);
                            NDArray classPredictions = output.get(1).softmax(-1).transpose(0, 2, 1);
                            NDArray boundingBoxPredictions = output.get(2);
                            MultiBoxDetection multiBoxDetection =
                                    new MultiBoxDetection.Builder().build();
                            NDList detections =
                                    multiBoxDetection.detection(
                                            new NDList(
                                                    classPredictions,
                                                    boundingBoxPredictions,
                                                    anchors));
                            return detections.singletonOrThrow().split(new int[] {1, 2}, 2);
                        }));
        return ssdPredict;
    }
}
