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

import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.examples.training.transferlearning.TrainResnetWithCifar10;
import ai.djl.training.TrainingResult;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class TrainResNetTest {
    private static final int SEED = 1234;

    @Test
    public void testTrainResNet() throws ModelException, IOException, TranslateException {
        // Limit max 4 gpu for cifar10 training to make it converge faster.
        // and only train 10 batch for unit test.
        String[] args = {"-e", "2", "-g", "4", "-m", "10", "-s", "-p"};

        TrainingResult result = TrainResnetWithCifar10.runExample(args);
        Assert.assertNotNull(result);
    }

    @Test
    public void testTrainResNetSymbolicNightly()
            throws ModelException, IOException, TranslateException {
        // this is nightly test
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("Nightly only");
        }
        if (Engine.getInstance().getGpuCount() > 0) {
            // Limit max 4 gpu for cifar10 training to make it converge faster.
            // and only train 10 batch for unit test.
            String[] args = {"-e", "10", "-g", "4", "-s", "-p"};

            Engine.getInstance().setRandomSeed(SEED);
            TrainingResult result = TrainResnetWithCifar10.runExample(args);
            Assert.assertNotNull(result);

            Assert.assertTrue(result.getTrainEvaluation("Accuracy") >= 0.8f);
            Assert.assertTrue(result.getValidateEvaluation("Accuracy") >= 0.68f);
            Assert.assertTrue(result.getValidateLoss() < 1.1);
        }
    }

    @Test
    public void testTrainResNetImperativeNightly()
            throws ModelException, IOException, TranslateException {
        // this is nightly test
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("Nightly only");
        }
        if (Engine.getInstance().getGpuCount() > 0) {
            // Limit max 4 gpu for cifar10 training to make it converge faster.
            // and only train 10 batch for unit test.
            String[] args = {"-e", "10", "-g", "4"};

            Engine.getInstance().setRandomSeed(SEED);
            TrainingResult result = TrainResnetWithCifar10.runExample(args);
            Assert.assertNotNull(result);

            Assert.assertTrue(result.getTrainEvaluation("Accuracy") >= 0.9f);
            Assert.assertTrue(result.getValidateEvaluation("Accuracy") >= 0.75f);
            Assert.assertTrue(result.getValidateLoss() < 1);
        }
    }
}
