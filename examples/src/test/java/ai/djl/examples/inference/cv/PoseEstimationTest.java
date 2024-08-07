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
package ai.djl.examples.inference.cv;

import ai.djl.ModelException;
import ai.djl.modality.cv.output.Joints;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class PoseEstimationTest {

    @Test
    public void testPoseEstimation() throws ModelException, TranslateException, IOException {
        Joints[] result = PoseEstimation.predict();
        Assert.assertEquals(result.length, 3);
        Assert.assertEquals(result[0].getJoints().size(), 17);
    }
}
