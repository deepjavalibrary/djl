/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.basicmodelzoo.cv.object_detection.yolo.YOLOV3;
import ai.djl.examples.training.util.Arguments;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.YOLOv3Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;

import java.io.IOException;

public final class TrainPikachuWithYOLOV3 {
    private TrainPikachuWithYOLOV3() {}

    public static void main(String[] args)
            throws IOException, TranslateException, MalformedModelException {
        TrainPikachuWithYOLOV3.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException, TranslateException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }

        try (Model model = Model.newInstance("pikachu-yolov3", arguments.getEngine())) {
            model.setBlock(YOLOV3.builder().setNumClasses(1).build());
            RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN, arguments);
            RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST, arguments);

            DefaultTrainingConfig config = setupTrainingConfig(arguments);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                Shape inputShape = new Shape(1, 3, 256, 256);
                trainer.initialize(inputShape);

                EasyTrain.fit(trainer, arguments.getEpoch(), trainingSet, validateSet);

                return trainer.getTrainingResult();
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
                        .setSampling(8, true)
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
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        float[] anchorsArray = YOLOv3Loss.getPresetAnchors();
        for (int i = 0; i < anchorsArray.length; i++) {
            anchorsArray[i] = anchorsArray[i] * 256 / 416; // reshaping into the
        }

        return new DefaultTrainingConfig(
                        YOLOv3Loss.builder()
                                .setNumClasses(1)
                                .setInputShape(new Shape(256, 256))
                                .setAnchorsArray(anchorsArray)
                                .build())
                .optDevices(arguments.getMaxGpus())
                .addTrainingListeners(TrainingListener.Defaults.basic())
                .addTrainingListeners(listener);
    }
}
