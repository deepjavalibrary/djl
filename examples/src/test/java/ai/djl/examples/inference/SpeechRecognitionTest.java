/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.engine.EngineException;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class SpeechRecognitionTest {

    @Test
    public void testSpeechRecognition() throws ModelException, TranslateException, IOException {
        TestRequirements.linux();
        TestRequirements.nightly();

        try {
            String result = SpeechRecognition.predict();
            Assert.assertEquals(
                    result,
                    "THE NEAREST SAID THE DISTRICT DOCTOR IS A GOOD ITALIAN ABBE WHO LIVES NEXT"
                            + " DOOR TO YOU SHALL I CALL ON HIM AS I PASS ");
        } catch (EngineException e) {
            // wav2vec2.ptl model requires avx2
            if (!"Unknown engine".equals(e.getMessage())) {
                throw e;
            }
        }
    }
}
