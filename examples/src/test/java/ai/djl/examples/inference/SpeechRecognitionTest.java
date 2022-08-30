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
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

import javax.sound.sampled.UnsupportedAudioFileException;

public class SpeechRecognitionTest {

    @Test
    public void testSpeechRecognition()
            throws ModelException, TranslateException, IOException, UnsupportedAudioFileException {
        TestRequirements.nightly();
        TestRequirements.engine("PyTorch");

        String result = SpeechRecognition.predict();
        Assert.assertEquals(
                result,
                "THE NEAREST SAID THE DISTRICT DOCTOR IS A GOOD ITALIAN ABBE WHO LIVES NEXT DOOR TO"
                        + " YOU SHALL I CALL ON HIM AS I PASS ");
    }
}
