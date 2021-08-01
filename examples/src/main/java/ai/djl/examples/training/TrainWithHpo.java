/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.examples.training.util.Arguments;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset.Usage;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.hyperparameter.EasyHpo;
import ai.djl.training.hyperparameter.param.HpInt;
import ai.djl.training.hyperparameter.param.HpSet;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class TrainWithHpo {

    private static final Logger logger = LoggerFactory.getLogger(TrainWithHpo.class);

    private TrainWithHpo() {}

    public static void main(String[] args) throws IOException, TranslateException {
        TrainWithHpo.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException, TranslateException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }

        return new Train(arguments).fit().getValue();
    }

    static class Train extends EasyHpo {

        Arguments arguments;

        public Train(Arguments arguments) {
            this.arguments = arguments;
        }

        @Override
        protected TrainingConfig setupTrainingConfig(HpSet hpVals) {
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

            return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    .addEvaluator(new Accuracy())
                    .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
                    .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                    .addTrainingListeners(listener);
        }

        @Override
        protected HpSet setupHyperParams() {
            return new HpSet(
                    "hp",
                    Arrays.asList(
                            new HpInt("hiddenLayersSize", 10, 100),
                            new HpInt("hiddenLayersCount", 2, 10)));
        }

        @Override
        protected RandomAccessDataset getDataset(Usage usage) throws IOException {
            Mnist mnist =
                    Mnist.builder()
                            .optUsage(usage)
                            .setSampling(arguments.getBatchSize(), true)
                            .optLimit(arguments.getLimit())
                            .build();
            mnist.prepare(new ProgressBar());
            return mnist;
        }

        @Override
        protected Model buildModel(HpSet hpVals) {
            int[] hidden = new int[(Integer) hpVals.getHParam("hiddenLayersCount").random()];
            Arrays.fill(hidden, (Integer) hpVals.getHParam("hiddenLayersSize").random());

            Block block =
                    new Mlp(Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH, Mnist.NUM_CLASSES, hidden);
            Model model = Model.newInstance("mlp");
            model.setBlock(block);
            return model;
        }

        @Override
        protected Shape inputShape(HpSet hpVals) {
            /*
             * MNIST is 28x28 grayscale image and pre processed into 28 * 28 NDArray.
             * 1st axis is batch axis, we can use 1 for initialization.
             */
            return new Shape(1, Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH);
        }

        @Override
        protected int numEpochs(HpSet hpVals) {
            return arguments.getEpoch();
        }

        @Override
        protected int numHyperParameterTests() {
            return 50;
        }

        @Override
        protected void saveModel(Model model, TrainingResult result) throws IOException {
            float loss = result.getValidateLoss();
            logger.info("--------- FINAL_HP - Loss {}", loss);

            model.setProperty("Epoch", String.valueOf(result.getEpoch()));
            model.setProperty(
                    "Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
            model.setProperty("Loss", String.format("%.5f", loss));
            model.save(Paths.get(arguments.getOutputDir()), "mlp");
        }
    }
}
