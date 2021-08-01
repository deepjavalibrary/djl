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

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.PikachuDetection;
import ai.djl.basicmodelzoo.cv.object_detection.ssd.SingleShotDetection;
import ai.djl.engine.Engine;
import ai.djl.examples.training.util.Arguments;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.MultiBoxDetection;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.BoundingBoxError;
import ai.djl.training.evaluator.SingleShotDetectionAccuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.SingleShotDetectionLoss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * An example of training a simple Single Shot Detection (SSD) model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/train_pikachu_ssd.md">doc</a>
 * for information about this example.
 */
public final class TrainPikachu {

    private TrainPikachu() {}

    public static void main(String[] args) throws IOException, TranslateException {
        TrainPikachu.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException, TranslateException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }

        try (Model model = Model.newInstance("pikachu-ssd")) {
            model.setBlock(getSsdTrainBlock());
            RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN, arguments);
            RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST, arguments);

            DefaultTrainingConfig config = setupTrainingConfig(arguments);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                Shape inputShape = new Shape(arguments.getBatchSize(), 3, 256, 256);
                trainer.initialize(inputShape);

                EasyTrain.fit(trainer, arguments.getEpoch(), trainingSet, validateSet);

                return trainer.getTrainingResult();
            }
        }
    }

    public static int predict(String outputDir, String imageFile)
            throws IOException, MalformedModelException, TranslateException {
        try (Model model = Model.newInstance("pikachu-ssd")) {
            float detectionThreshold = 0.6f;
            // load parameters back to original training block
            model.setBlock(getSsdTrainBlock());
            model.load(Paths.get(outputDir));
            // append prediction logic at end of training block with parameter loaded
            Block ssdTrain = model.getBlock();
            model.setBlock(getSsdPredictBlock(ssdTrain));
            Path imagePath = Paths.get(imageFile);
            SingleShotDetectionTranslator translator =
                    SingleShotDetectionTranslator.builder()
                            .addTransform(new ToTensor())
                            .optSynset(Collections.singletonList("pikachu"))
                            .optThreshold(detectionThreshold)
                            .build();
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor(translator)) {
                Image image = ImageFactory.getInstance().fromFile(imagePath);
                DetectedObjects detectedObjects = predictor.predict(image);
                image.drawBoundingBoxes(detectedObjects);
                Path out = Paths.get(outputDir).resolve("pikachu_output.png");
                image.save(Files.newOutputStream(out), "png");
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
                        .optLimit(arguments.getLimit())
                        .optPipeline(pipeline)
                        .setSampling(arguments.getBatchSize(), true)
                        .build();
        pikachuDetection.prepare(new ProgressBar());

        return pikachuDetection;
    }

    private static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        String outputDir = arguments.getOutputDir();
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("classAccuracy");
                    model.setProperty("ClassAccuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        return new DefaultTrainingConfig(new SingleShotDetectionLoss())
                .addEvaluator(new SingleShotDetectionAccuracy("classAccuracy"))
                .addEvaluator(new BoundingBoxError("boundingBoxError"))
                .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
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
