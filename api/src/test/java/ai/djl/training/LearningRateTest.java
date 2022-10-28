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

import ai.djl.training.tracker.CosineTracker;
import ai.djl.training.tracker.CyclicalTracker;
import ai.djl.training.tracker.FactorTracker;
import ai.djl.training.tracker.FixedPerVarTracker;
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

    @Test
    public void testCosineTracker() {
        float baseValue = 0.5f;
        float finalValue = 0.01f;
        double epsilon = 1e-3;
        CosineTracker cosineTracker =
                Tracker.cosine()
                        .setBaseValue(baseValue)
                        .optFinalValue(finalValue)
                        .setMaxUpdates(200)
                        .build();

        Assert.assertEquals(cosineTracker.getNewValue(0), baseValue);
        Assert.assertEquals(cosineTracker.getNewValue(20), 0.488f, epsilon);
        Assert.assertEquals(cosineTracker.getNewValue(50), 0.428f, epsilon);
        Assert.assertEquals(cosineTracker.getNewValue(100), 0.255f, epsilon);
        Assert.assertEquals(cosineTracker.getNewValue(150), 0.082f, epsilon);
        Assert.assertEquals(cosineTracker.getNewValue(180), 0.022f, epsilon);
        Assert.assertEquals(cosineTracker.getNewValue(200), finalValue);
        Assert.assertEquals(cosineTracker.getNewValue(300), finalValue);
    }

    @Test
    public void testCyclicalTracker() {
        float baseValue = 0.01f;
        float maxValue = 0.1f;
        double epsilon = 1e-3;
        CyclicalTracker cyclicalTrackerTriangular =
                Tracker.cyclical()
                        .optBaseValue(baseValue)
                        .optMaxValue(maxValue)
                        .optMode(CyclicalTracker.CyclicalMode.TRIANGULAR)
                        .optStepSizeUp(100)
                        .build();
        Assert.assertEquals(cyclicalTrackerTriangular.getNewValue(0), baseValue);
        Assert.assertEquals(cyclicalTrackerTriangular.getNewValue(10), 0.019f, epsilon);
        Assert.assertEquals(cyclicalTrackerTriangular.getNewValue(50), 0.055f, epsilon);
        Assert.assertEquals(cyclicalTrackerTriangular.getNewValue(100), 0.1f, epsilon);
        Assert.assertEquals(cyclicalTrackerTriangular.getNewValue(380), 0.028f, epsilon);

        CyclicalTracker cyclicalTrackerUnbalance =
                Tracker.cyclical()
                        .optBaseValue(baseValue)
                        .optMaxValue(maxValue)
                        .optMode(CyclicalTracker.CyclicalMode.TRIANGULAR)
                        .optStepSizeUp(100)
                        .optStepSizeDown(50)
                        .build();
        Assert.assertEquals(cyclicalTrackerUnbalance.getNewValue(0), baseValue);
        Assert.assertEquals(cyclicalTrackerUnbalance.getNewValue(10), 0.019f, epsilon);
        Assert.assertEquals(cyclicalTrackerUnbalance.getNewValue(50), 0.055f, epsilon);
        Assert.assertEquals(cyclicalTrackerUnbalance.getNewValue(130), 0.046f, epsilon);
        Assert.assertEquals(cyclicalTrackerUnbalance.getNewValue(380), 0.082f, epsilon);

        CyclicalTracker cyclicalTrackerTriangular2 =
                Tracker.cyclical()
                        .optBaseValue(baseValue)
                        .optMaxValue(maxValue)
                        .optMode(CyclicalTracker.CyclicalMode.TRIANGULAR2)
                        .optStepSizeUp(100)
                        .build();
        Assert.assertEquals(cyclicalTrackerTriangular2.getNewValue(0), baseValue);
        Assert.assertEquals(cyclicalTrackerTriangular2.getNewValue(50), 0.055f, epsilon);
        Assert.assertEquals(cyclicalTrackerTriangular2.getNewValue(100), 0.1f, epsilon);
        Assert.assertEquals(cyclicalTrackerTriangular2.getNewValue(380), 0.019f, epsilon);
        Assert.assertEquals(cyclicalTrackerTriangular2.getNewValue(460), 0.0235f, epsilon);

        CyclicalTracker cyclicalTrackerExpRange =
                Tracker.cyclical()
                        .optBaseValue(baseValue)
                        .optMaxValue(maxValue)
                        .optMode(CyclicalTracker.CyclicalMode.EXP_RANGE)
                        .optGamma(0.999f)
                        .optStepSizeUp(100)
                        .build();
        Assert.assertEquals(cyclicalTrackerExpRange.getNewValue(0), baseValue);
        Assert.assertEquals(cyclicalTrackerExpRange.getNewValue(50), 0.0528f, epsilon);
        Assert.assertEquals(cyclicalTrackerExpRange.getNewValue(380), 0.0223f, epsilon);

        CyclicalTracker cyclicalTrackerCustom =
                Tracker.cyclical()
                        .optBaseValue(baseValue)
                        .optMaxValue(maxValue)
                        .optScaleFunction(new CustomScaleFunction())
                        .optScaleModeCycle(true)
                        .optStepSizeUp(100)
                        .build();
        Assert.assertEquals(cyclicalTrackerCustom.getNewValue(0), baseValue);
        Assert.assertEquals(cyclicalTrackerCustom.getNewValue(50), 0.055f, epsilon);
        Assert.assertEquals(cyclicalTrackerCustom.getNewValue(500), 0.04f, epsilon);
    }

    @Test
    public void testFixedPerVarTracker() {
        float lr = 0.001f; // Customized learning rate
        FixedPerVarTracker.Builder fixedPerVarTrackerBuilder =
                FixedPerVarTracker.builder().setDefaultValue(lr);
        String[] parameterIds = {"id1", "id2"};
        for (String param : parameterIds) {
            fixedPerVarTrackerBuilder.put(param, 0.1f * lr);
        }
        FixedPerVarTracker fixedPerVarTracker = fixedPerVarTrackerBuilder.build();

        Assert.assertEquals(fixedPerVarTracker.getNewValue("id1", 0), 0.1f * lr);
        Assert.assertEquals(fixedPerVarTracker.getNewValue("id2", 1), 0.1f * lr);
        Assert.assertEquals(fixedPerVarTracker.getNewValue("unknown", 2), lr);
    }

    private static class CustomScaleFunction implements CyclicalTracker.ScaleFunction {
        @Override
        public float func(int steps) {
            if (steps == 0) {
                return 1;
            }
            return 1f / ((float) steps);
        }
    }
}
