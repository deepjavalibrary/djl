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
import ai.djl.engine.Engine;
import ai.djl.testing.TestRequirements;
import ai.djl.training.TrainingResult;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class TrainPikachuTest {

    @Test
    public void testDetection() throws IOException, MalformedModelException, TranslateException {
        TestRequirements.linux();
        TestRequirements.nightly();

        String[] args;
        float expectedLoss = 0;
        int expectedMinNumber = 0;
        int expectedMaxNumber = 0;
        // TODO: implement PyTorch multiBoxTarget or object detection training
        if (Engine.getEngine("MXNet").getGpuCount() > 0) {
            args = new String[] {"-e", "20", "-b", "32", "-g", "1", "--engine", "MXNet"};
            expectedLoss = 2.5e-3f;
            expectedMaxNumber = 15;
            expectedMinNumber = 6;
        } else {
            // test train 1 epoch and predict workflow works on CPU
            args = new String[] {"-e", "1", "-m", "1", "-b", "32", "--engine", "MXNet"};
        }
        // test train
        TrainingResult result = TrainPikachu.runExample(args);
        Assert.assertNotNull(result);

        if (expectedLoss > 0) {
            Assert.assertTrue(result.getValidateLoss() < expectedLoss);
        }

        // test predict
        int numberOfPikachus =
                TrainPikachu.predict("build/model", "src/test/resources/pikachu.jpg");
        if (expectedMinNumber > 0) {
            Assert.assertTrue(numberOfPikachus >= expectedMinNumber);
            Assert.assertTrue(numberOfPikachus <= expectedMaxNumber);
        }
    }
}
