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
import ai.djl.modality.cv.transform.OneHot;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.transform.Transpose;
import ai.djl.ndarray.NDArray;
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

public final class TransferFreshFruit {

    private TransferFreshFruit() {}

    public static void main(String[] args)
            throws IOException, TranslateException, ModelException, URISyntaxException {
        TransferFreshFruit.runExample(args);
    }

    public static TrainingResult runExample(String[] args)
            throws IOException, TranslateException, ModelException, URISyntaxException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }

        // Also available at
        // "https://mlrepo.djl.ai/model/cv/image_classification/ai/djl/pytorch/resnet18_embedding/0.0.1/traced_resnet18_embedding.pt.gz";
        String modelUrls = "/Users/fenkexin/Desktop/transferDJL/code/base_nw.pt";
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(modelUrls)
                        .optEngine(Engine.getDefaultEngineName())
                        .optProgress(new ProgressBar())
                        // Here the argument pretrained is borrowed.
                        // Pretrained means no need to retrain, and vice versa.
                        .optOption("retrain", arguments.isPreTrained() ? "0" : "1")
                        .build();

        ZooModel<NDList, NDList> embedding = criteria.loadModel();

        Block baseBlock = embedding.getBlock();
        Block blocks =
                new SequentialBlock()
                        .add(baseBlock)
                        .addSingleton(nd -> nd.squeeze(new int[] {2, 3}))
                        .add(Linear.builder().setUnits(2).build()) // linear on which dim?
                        .addSingleton(nd -> nd.softmax(1));

        Model model = Model.newInstance("TransferFreshFruit");
        model.setBlock(blocks);

        // Config trainer
        DefaultTrainingConfig config = setupTrainingConfig(arguments);

        // Customized learning rate
        float lr = 0.001f;
        FixedPerVarTracker.Builder learningRateTrackerBuilder =
                FixedPerVarTracker.builder().setDefaultValue(lr);
        for (Pair<String, Parameter> paramPair : baseBlock.getParameters()) {
            learningRateTrackerBuilder.put(paramPair.getValue().getId(), 0.1f * lr);
        }
        Optimizer optimizer =
                Adam.builder().optLearningRateTracker(learningRateTrackerBuilder.build()).build();
        config.optOptimizer(optimizer);

        // Config trainer
        Trainer trainer = model.newTrainer(config);
        trainer.setMetrics(new Metrics());

        // Initialize the parameter shape and value
        int batchSize = 32;
        Shape inputShape = new Shape(batchSize, 3, 224, 224);
        trainer.initialize(inputShape);

        // Data
        ImageFolder datasetTrain = getData("test", "banana", batchSize);

        // Train
        EasyTrain.fit(trainer, 6, datasetTrain, null);

        // Save model
        // model.save("your-model-path");

        model.close();
        embedding.close();
        return null;
    }

    private static ImageFolder getData(String subfolderName, String fruit, int batchSize)
            throws TranslateException, IOException {
        // The dataset is from <a
        // href="https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification">https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification</a>
        String folderUrl = "/Users/fenkexin/Desktop/transferDJL/code/data/" + fruit;
        String subfolder = "/" + subfolderName + "/";
        Repository repository = Repository.newInstance("banana", Paths.get(folderUrl + subfolder));
        ImageFolder dataset =
                ImageFolder.builder()
                        .setRepository(repository)
                        .addTransform(new ToTensor())
                        .addTransform(new Transpose(1, 2, 0))
                        .addTransform(new Resize(224, 224))
                        .addTransform(new Transpose(2, 0, 1))
                        .addTargetTransform(new OneHot(2))
                        .setSampling(batchSize, true)
                        .build();
        dataset.prepare();
        return dataset;
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

        return new DefaultTrainingConfig(new SoftmaxCrossEntropy("SoftmaxCrossEntropy"))
                .addEvaluator(new Accuracy())
                .optDevices(Engine.getInstance().getDevices(1))
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }

    private static class SoftmaxCrossEntropy extends Loss {

        /**
         * Base class for metric with abstract update methods.
         *
         * @param name The display name of the Loss
         */
        public SoftmaxCrossEntropy(String name) {
            super(name);
        }

        /** {@inheritDoc} */
        @Override
        public NDArray evaluate(NDList labels, NDList predictions) {
            // Here the labels are supposed to be one-hot
            int classAxis = -1;
            NDArray pred = predictions.singletonOrThrow().log();
            NDArray lab = labels.singletonOrThrow().reshape(pred.getShape());
            NDArray loss = pred.mul(lab).neg().sum(new int[] {classAxis}, true);
            return loss.mean();
        }
    }
}
