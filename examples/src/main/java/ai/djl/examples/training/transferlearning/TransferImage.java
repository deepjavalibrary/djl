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
package ai.djl.examples.training.transferlearning;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.engine.Engine;
import ai.djl.examples.training.util.Arguments;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.FixedPerVarTracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Paths;

public final class TransferImage {

    private TransferImage() {}

    public static void main(String[] args)
            throws IOException, TranslateException, ModelException, URISyntaxException {
        TransferImage.runExample(args);
    }

    public static TrainingResult runExample(String[] args)
            throws IOException, TranslateException, ModelException, URISyntaxException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }
        System.out.println(Engine.getDefaultEngineName());

        String modelUrls = "/Users/fenkexin/Desktop/transferDJL/code/base_nw.pt";
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(modelUrls)
                        .optEngine(Engine.getDefaultEngineName())
                        .optProgress(new ProgressBar())
                        .optOption("retrain", "1")
                        .build();

        ZooModel<NDList, NDList> embedding = criteria.loadModel();

        Block baseBlock = embedding.getBlock();
        Block blocks =
                new SequentialBlock()
                        .add(baseBlock)
                        .addSingleton(nd -> nd.squeeze(new int[] {2, 3}))
                        .add(Linear.builder().setUnits(2).build()) // linear on which dim?
                        .addSingleton(nd -> nd.softmax(1));

        Model model = Model.newInstance("TransferImage");
        model.setBlock(blocks);

        // Config trainer
        DefaultTrainingConfig config = setupTrainingConfig(arguments);

        /// Customized learning rate
        FixedPerVarTracker.Builder learningRateTrackerBuilder =
                FixedPerVarTracker.builder().setDefaultValue(0.001f);
        for (Pair<String, Parameter> paramPair : baseBlock.getParameters()) {
            learningRateTrackerBuilder.put(paramPair.getValue().getId(), 0.0001f);
        }
        Optimizer optimizer =
                Adam.builder().optLearningRateTracker(learningRateTrackerBuilder.build()).build();
        config.optOptimizer(optimizer);

        Trainer trainer = model.newTrainer(config);
        trainer.setMetrics(new Metrics());

        int batchSize = 32;
        Shape inputShape = new Shape(batchSize, 3, 224, 224);

        // initialize trainer with proper input shape
        trainer.initialize(inputShape);

        // Data
        String folderUrl = "/Users/fenkexin/Desktop/transferDJL/code/data/banana";
        String subfolder = "/test/";
        Repository repository = Repository.newInstance("banana", Paths.get(folderUrl + subfolder));
        ImageFolder datasetTrain =
                ImageFolder.builder()
                        .setRepository(repository)
                        .addTransform(new ToTensor())
                        .addTransform(new Resize(224, 224))
                        //                        .addTargetTransform(new ToOneHot(2))
                        .setSampling(batchSize, true)
                        .build();
        datasetTrain.prepare();

        // train
        EasyTrain.fit(trainer, 50, datasetTrain, null);

        // Save model
        // model.save("your-model-path");

        return null;
    }

    private static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        String outputDir = arguments.getOutputDir();
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("Accuracy");
                    model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss("SoftmaxCrossEntropy"))
                .addEvaluator(new Accuracy())
                .optDevices(Engine.getInstance().getDevices(1))
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }
}
