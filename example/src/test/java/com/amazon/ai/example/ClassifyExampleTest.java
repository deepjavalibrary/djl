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
package com.amazon.ai.example;

import com.amazon.ai.example.util.AbstractExample;
import com.amazon.ai.example.util.ModelInfo;
import com.amazon.ai.inference.Classification;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ClassifyExampleTest {

    private static final String MODEL_NAME = "squeezenet_v1.1";

    @Test
    public void testGenericeInferenceExample() {
        ModelInfo modelInfo = ModelInfo.getModel(MODEL_NAME);
        String[] args =
                new String[] {
                    "-u",
                    modelInfo.getUrl(),
                    "-n",
                    MODEL_NAME,
                    "-i",
                    "src/test/resources/kitten.jpg",
                    "-c",
                    "1",
                    "-l",
                    "build/logs"
                };
        Assert.assertTrue(new ClassifyExample().runExample(args));
        Classification result = (Classification) AbstractExample.getPredictResult();
        Assert.assertEquals(result.getClassName(), "tabby, tabby cat");
        Assert.assertTrue(Double.compare(result.getProbability(), 0.7) > 0);
    }
}
