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
import ai.djl.examples.inference.ImageClassification;
import ai.djl.modality.Classifications;
import ai.djl.testing.TestRequirements;
import ai.djl.training.TrainingResult;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class TrainMnistTest {

    @Test
    public void testTrainMnist() throws ModelException, TranslateException, IOException {
        TestRequirements.engine("MXNet", "PyTorch");

        double expectedProb;
        if (Boolean.getBoolean("nightly")) {
            String[] args = new String[] {"-g", "1"};

            TrainingResult result = TrainMnist.runExample(args);
            Assert.assertNotNull(result);

            float accuracy = result.getValidateEvaluation("Accuracy");
            float loss = result.getValidateLoss();
            Assert.assertTrue(accuracy > 0.9f, "Accuracy: " + accuracy);
            Assert.assertTrue(loss < 0.35f, "Loss: " + loss);

            expectedProb = 0.9;
        } else {
            String[] args = new String[] {"-g", "1", "-m", "2"};

            TrainMnist.runExample(args);
            expectedProb = 0;
        }

        Classifications classifications = ImageClassification.predict();
        Classifications.Classification best = classifications.best();
        if (Boolean.getBoolean("nightly")) {
            Assert.assertEquals(best.getClassName(), "0");
        }
        double probability = best.getProbability();
        Assert.assertTrue(probability > expectedProb && probability <= 1);
    }
}
