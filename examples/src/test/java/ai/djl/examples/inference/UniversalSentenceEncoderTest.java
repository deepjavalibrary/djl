/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class UniversalSentenceEncoderTest {

    @Test
    public void testSentimentAnalysis() throws ModelException, TranslateException, IOException {
        TestRequirements.linux();
        TestRequirements.nightly();

        List<String> inputs = new ArrayList<>();
        inputs.add("The quick brown fox jumps over the lazy dog.");
        inputs.add("I am a sentence for which I would like to get its embedding");

        float[][] result = UniversalSentenceEncoder.predict(inputs);
        Assert.assertNotNull(result);

        Assert.assertEquals(result[0][0], -0.031330183, 0.0001);
    }
}
