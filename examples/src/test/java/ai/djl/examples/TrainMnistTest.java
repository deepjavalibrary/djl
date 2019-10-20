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
package ai.djl.examples;

import ai.djl.examples.training.TrainMnist;
import org.testng.Assert;
import org.testng.annotations.Test;

public class TrainMnistTest {

    @Test
    public void testTrainMnist() {
        String[] args = new String[] {"-e", "2", "-o", "build/logs"};

        TrainMnist test = new TrainMnist();
        Assert.assertTrue(test.runExample(args));
        Assert.assertTrue(test.getTrainingAccuracy() > 0.9f);
        Assert.assertTrue(test.getTrainingLoss() < 0.2f);
    }
}
