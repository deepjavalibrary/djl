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

import ai.djl.examples.inference.ClassifyExample;
import ai.djl.examples.inference.util.AbstractExample;
import ai.djl.modality.Classification;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ClassifyExampleTest {

    @Test
    public void testClassifyExample() {
        String[] args = {"-i", "src/test/resources/kitten.jpg", "-c", "1", "-l", "build/logs"};
        Assert.assertTrue(new ClassifyExample().runExample(args));
        Classification result = (Classification) AbstractExample.getPredictResult();
        Classification.Item best = result.best();
        Assert.assertEquals(best.getClassName(), "n02123045 tabby, tabby cat");
        Assert.assertTrue(Double.compare(best.getProbability(), 0.4) > 0);
    }
}
