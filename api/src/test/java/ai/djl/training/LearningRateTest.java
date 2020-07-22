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

import ai.djl.training.tracker.FactorTracker;
import ai.djl.training.tracker.MultiFactorTracker;
import ai.djl.training.tracker.Tracker;
import org.testng.Assert;
import org.testng.annotations.Test;

public class LearningRateTest {

    @Test
    public void testFactorTracker() {

        FactorTracker factorTracker = Tracker.factor().setFactor(0.5f).setBaseValue(1f).build();
        Assert.assertEquals(factorTracker.getNewValue(0), 1f);
        Assert.assertEquals(factorTracker.getNewValue(1), 0.5f);
        Assert.assertEquals(factorTracker.getNewValue(3), 0.125f);
    }

    @Test
    public void testMultiFactorTracker() {

        MultiFactorTracker factorTracker =
                Tracker.multiFactor()
                        .setSteps(new int[] {100, 250, 500})
                        .optFactor(0.5f)
                        .setBaseValue(1f)
                        .build();
        Assert.assertEquals(factorTracker.getNewValue(1), 1f);
        Assert.assertEquals(factorTracker.getNewValue(100), 1f);
        Assert.assertEquals(factorTracker.getNewValue(101), .5f);
        Assert.assertEquals(factorTracker.getNewValue(250), .5f);
        Assert.assertEquals(factorTracker.getNewValue(251), .25f);
        Assert.assertEquals(factorTracker.getNewValue(500), .25f);
        Assert.assertEquals(factorTracker.getNewValue(501), .125f);
    }
}
