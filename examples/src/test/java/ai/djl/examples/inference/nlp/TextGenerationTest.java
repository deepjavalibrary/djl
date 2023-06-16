/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference.nlp;

import ai.djl.ModelException;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class TextGenerationTest {

    @Test
    public void testTextGeneration() throws TranslateException, ModelException, IOException {
//        TestRequirements.nightly();
//        TestRequirements.engine("PyTorch");

        String expected =
                "DeepMind Company is a global leader in the field of artificial"
                        + " intelligence and artificial intelligence. We are a leading provider"
                        + " of advanced AI solutions for the automotive industry, including the"
                        + " latest in advanced AI solutions for the automotive industry. We are"
                        + " also a leading provider of advanced AI solutions for the automotive"
                        + " industry, including the";

        Assert.assertEquals(TextGeneration.generateTextWithPyTorch(), expected);
    }
}
