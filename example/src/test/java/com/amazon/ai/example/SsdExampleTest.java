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
import com.amazon.ai.inference.DetectedObject;
import java.io.IOException;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class SsdExampleTest {

    private static final String MODEL_NAME = "resnet50_ssd_model";

    private ModelInfo modelInfo = ModelInfo.getModel(MODEL_NAME);

    @BeforeClass
    public void downloadModel() throws IOException {
        modelInfo.download();
    }

    @Test
    public void testSsdExample() {
        Path dir = modelInfo.getDownloadDir();
        String[] args =
                new String[] {
                    "-p",
                    dir.toString(),
                    "-n",
                    MODEL_NAME,
                    "-i",
                    "src/test/resources/3dogs.jpg",
                    "-c",
                    "1",
                    "-l",
                    "build/logs"
                };
        Assert.assertTrue(new SsdExample().runExample(args));
        DetectedObject result = (DetectedObject) AbstractExample.getPredictResult();
        Assert.assertEquals(result.getClassName(), "dog");
        Assert.assertTrue(Double.compare(result.getProbability(), 0.8) > 0);
    }
}
