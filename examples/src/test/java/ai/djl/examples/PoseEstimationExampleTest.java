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

import ai.djl.examples.inference.PoseEstimationExample;
import ai.djl.modality.cv.Joints;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PoseEstimationExampleTest {

    @Test
    public void testPoseEstimation() {
        String[] args = {"-i", "src/test/resources/pose_soccer.png", "-c", "1", "-l", "build/logs"};

        PoseEstimationExample test = new PoseEstimationExample();
        Assert.assertTrue(test.runExample(args));
        List<Joints> result = test.getPredictResult();
        Joints current = result.get(0);
        Assert.assertEquals(result.size(), 3);
        Assert.assertTrue(current.getJoints().get(0).getConfidence() > 0.6d);
    }
}
