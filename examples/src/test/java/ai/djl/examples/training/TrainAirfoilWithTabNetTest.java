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

import ai.djl.testing.TestRequirements;
import ai.djl.training.TrainingResult;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class TrainAirfoilWithTabNetTest {

    @Test
    public void testTrainAirfoilWithTabNet() throws TranslateException, IOException {
        TestRequirements.linux(); // TODO: NaN on Windows for PyTorch 2.3.1

        String[] args = {"-g", "1", "-e", "20", "-b", "32"};
        if (!Boolean.getBoolean("nightly")) {
            args[3] = "2";
        }
        TrainingResult result = TrainAirfoilWithTabNet.runExample(args);
        Assert.assertNotNull(result);
        float loss = result.getValidateLoss();
        if (Boolean.getBoolean("nightly")) {
            Assert.assertTrue(loss < 50f, "Loss: " + loss);
        }
    }
}
