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

import ai.djl.examples.inference.ActionRecognition;
import ai.djl.examples.inference.util.AbstractExample;
import ai.djl.modality.Classification;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ActionRecognitionExampleTest {

    @Test
    public void testActionRecognition() {
        String[] args = {
            "-i", "src/test/resources/action_discus_throw.png", "-c", "1", "-l", "build/logs"
        };
        Assert.assertTrue(new ActionRecognition().runExample(args));
        Classification result = (Classification) AbstractExample.getPredictResult();
        Classification.Item best = result.best();
        Assert.assertEquals(best.getClassName(), "ThrowDiscus");
        Assert.assertTrue(Double.compare(best.getProbability(), 0.9) > 0);
    }
}
