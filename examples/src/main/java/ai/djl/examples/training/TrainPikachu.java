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
import ai.djl.basicmodelzoo.cv.object_detection.ssd.SingleShotDetection;
import ai.djl.examples.training.util.Arguments;
import ai.djl.examples.training.util.ExampleTrainingResult;
import ai.djl.examples.training.util.TrainingUtils;
import ai.djl.inference.Predictor;
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
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.BoundingBoxError;
import ai.djl.training.evaluator.SingleShotDetectionAccuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.SingleShotDetectionLoss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.imageio.ImageIO;
import org.apache.commons.cli.ParseException;

/**
 * An example of training a simple Single Shot Detection (SSD) model.
 *
 * <p>See this <a
 * href="https://github.com/awslabs/djl/blob/master/examples/docs/train_pikachu_ssd.md">doc</a> for
 * information about this example.
 */
public final class TrainPikachu {

    private TrainPikachu() {}

    public static void main(String[] args) throws IOException, ParseException {
        TrainPikachu.runExample(args);
    }

    public static ExampleTrainingResult runExample(String[] args)
            throws IOException, ParseException {
        Arguments arguments = Arguments.parseArgs(args);

        try (Model model = Model.newInstance()) {
            model.setBlock(getSsdTrainBlock());

            RandomAccessDataset pikachuDetectionTrain = getDataset(Dataset.Usage.TRAIN, arguments);
            RandomAccessDataset pikachuDetectionTest = getDataset(Dataset.Usage.TEST, arguments);

            DefaultTrainingConfig config = setupTrainingConfig(arguments);
            config.addTrainingListeners(
                    TrainingListener.Defaults.logging(
                            TrainPikachu.class.getSimpleName(),
                            arguments.getBatchSize(),
                            (int) pikachuDetectionTrain.getNumIterations(),
                            (int) pikachuDetectionTest.getNumIterations(),
                            arguments.getOutputDir()));

            ExampleTrainingResult result;
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                Shape inputShape = new Shape(arguments.getBatchSize(), 3, 256, 256);
                trainer.initialize(inputShape);
                TrainingUtils.fit(
                        trainer,
                        arguments.getEpoch(),
                        pikachuDetectionTrain,
                        pikachuDetectionTest,
                        arguments.getOutputDir(),
                        "ssd");

                result = new ExampleTrainingResult(trainer);
            }
            model.save(Paths.get(arguments.getOutputDir()), "ssd");
            return result;
        }
    }

    public static int predict(String outputDir, String imageFile)
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
                    SingleShotDetectionTranslator.builder()
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

    private static RandomAccessDataset getDataset(Dataset.Usage usage, Arguments arguments)
            throws IOException {
        Pipeline pipeline = new Pipeline(new ToTensor());
        PikachuDetection pikachuDetection =
                PikachuDetection.builder()
                        .optUsage(usage)
                        .optMaxIteration(arguments.getMaxIterations())
                        .optPipeline(pipeline)
                        .setSampling(arguments.getBatchSize(), true)
                        .build();
        pikachuDetection.prepare(new ProgressBar());

        return pikachuDetection;
    }

    private static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        return new DefaultTrainingConfig(new SingleShotDetectionLoss())
                .setBatchSize(arguments.getBatchSize())
                .addEvaluator(new SingleShotDetectionAccuracy("classAccuracy"))
                .addEvaluator(new BoundingBoxError("boundingBoxError"))
                .optDevices(Device.getDevices(arguments.getMaxGpus()));
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

        return SingleShotDetection.builder()
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
                                    MultiBoxDetection.builder().build();
                            NDList detections =
                                    multiBoxDetection.detection(
                                            new NDList(
                                                    classPredictions,
                                                    boundingBoxPredictions,
                                                    anchors));
                            return detections.singletonOrThrow().split(new long[] {1, 2}, 2);
                        }));
        return ssdPredict;
    }
}
