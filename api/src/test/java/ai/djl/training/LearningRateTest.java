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

package ai.djl.training;

import ai.djl.training.optimizer.learningrate.FactorTracker;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.training.optimizer.learningrate.MultiFactorTracker;
import org.testng.Assert;
import org.testng.annotations.Test;

public class LearningRateTest {

    @Test
    public void testFactorTracker() {

        FactorTracker factorTracker =
                LearningRateTracker.factorTracker()
                        .setStep(250)
                        .optFactor(0.5f)
                        .optBaseLearningRate(1f)
                        .build();
        Assert.assertEquals(factorTracker.getNewLearningRate(1), 1f);
        Assert.assertEquals(factorTracker.getNewLearningRate(250), 1f);
        Assert.assertEquals(factorTracker.getNewLearningRate(251), .5f);
        Assert.assertEquals(factorTracker.getNewLearningRate(500), .5f);
        Assert.assertEquals(factorTracker.getNewLearningRate(501), .25f);
    }

    @Test
    public void testMultiFactorTracker() {

        MultiFactorTracker factorTracker =
                LearningRateTracker.multiFactorTracker()
                        .setSteps(new int[] {100, 250, 500})
                        .optFactor(0.5f)
                        .optBaseLearningRate(1f)
                        .build();
        Assert.assertEquals(factorTracker.getNewLearningRate(1), 1f);
        Assert.assertEquals(factorTracker.getNewLearningRate(100), 1f);
        Assert.assertEquals(factorTracker.getNewLearningRate(101), .5f);
        Assert.assertEquals(factorTracker.getNewLearningRate(250), .5f);
        Assert.assertEquals(factorTracker.getNewLearningRate(251), .25f);
        Assert.assertEquals(factorTracker.getNewLearningRate(500), .25f);
        Assert.assertEquals(factorTracker.getNewLearningRate(501), .125f);
    }
}
