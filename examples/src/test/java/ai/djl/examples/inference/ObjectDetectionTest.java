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
package ai.djl.examples.inference;

import ai.djl.ModelException;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class ObjectDetectionTest {

    private static final Logger logger = LoggerFactory.getLogger(ObjectDetectionTest.class);

    @Test
    public void testObjectDetection() throws ModelException, TranslateException, IOException {
        TestRequirements.engine("MXNet", "PyTorch", "TensorFlow");

        DetectedObjects result = ObjectDetection.predict();
        logger.info("{}", result);

        Assert.assertTrue(result.getNumberOfObjects() >= 3);
        Classifications.Classification obj = result.best();
        String className = obj.getClassName();
        List<String> objects = Arrays.asList("dog", "Cat", "car");
        Assert.assertTrue(objects.contains(className));
        Assert.assertTrue(obj.getProbability() > 0.6);
    }
}
