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
package ai.djl.integration.tests.inference;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.integration.util.TestUtils;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.testing.TestRequirements;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.loss.Loss;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

public class PredictorTest {

    @Test
    public void testPredictorWithDeviceGpu() throws TranslateException {
        TestRequirements.gpu();

        // CPU model, GPU predictor
        predictWithDeviceHelper(Device.cpu(), Device.gpu());
    }

    @Test
    public void testPredictorWithDeviceCpu() throws TranslateException {
        TestRequirements.gpu();

        // GPU model, CPU predictor
        predictWithDeviceHelper(Device.gpu(), Device.cpu());
    }

    public void predictWithDeviceHelper(Device device, Device predictorDevice)
            throws TranslateException {
        // Create simple model on modelDevice
        try (NDManager manager = NDManager.newBaseManager(device, TestUtils.getEngine());
                Model model = Model.newInstance("mlp", device, TestUtils.getEngine())) {
            Block block = new Mlp(10, 10, new int[] {10});
            model.setBlock(block);

            // Use trainer for initialization
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                            .optDevices(new Device[] {device});
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, 10));
            }

            // Create predictor on predictorDevice
            try (Predictor<NDList, NDList> predictor =
                    model.newPredictor(new NoopTranslator(null), predictorDevice)) {

                // Pass in predictorDevice input
                try (NDManager gpuManager = manager.newSubManager(predictorDevice)) {
                    NDList input = new NDList(gpuManager.ones(new Shape(1, 10)));

                    // The result should still be on predictorDevice
                    NDList result = predictor.predict(input);
                    Assert.assertEquals(result.get(0).getDevice(), predictorDevice);
                }
            }
        }
    }
}
