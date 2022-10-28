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

import ai.djl.ModelException;
import ai.djl.examples.training.transferlearning.TransferFreshFruit;
import ai.djl.testing.TestRequirements;
import ai.djl.training.TrainingResult;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.URISyntaxException;

public class TransferFreshFruitTest {

    @Test
    public void testTransferFreshFruit()
            throws ModelException, TranslateException, IOException, URISyntaxException {
        TestRequirements.engine("PyTorch");

        String[][] args = {{}, {"-p"}};
        TrainingResult result;
        for (String[] arg : args) {
            result = TransferFreshFruit.runExample(arg);
            Assert.assertNotNull(result);
            Assert.assertTrue(result.getEvaluations().get("validate_Accuracy") > 0.9f);
            Assert.assertTrue(result.getEvaluations().get("train_Accuracy") > 0.9f);
        }
    }
}
