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
package ai.djl.llama.engine;

import ai.djl.llama.engine.LlamaTranslator.TextGenerationInput;
import ai.djl.llama.jni.InputParameters;
import ai.djl.util.JsonUtils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

public class InputParametersTest {

    @Test
    public void testInputParameters() throws IOException {
        Path file = Paths.get("src/test/resources/inputs.json");
        try (Reader reader = Files.newBufferedReader(file)) {
            TextGenerationInput in = JsonUtils.GSON.fromJson(reader, TextGenerationInput.class);
            InputParameters param = in.parameters.toInputParameters();
            Assert.assertEquals(param.getMaxNewTokens(), 2);
            Assert.assertEquals(param.getNumberKeep(), 2);
            Assert.assertEquals(param.getNumberProbabilities(), 2);
            Assert.assertEquals(param.getTopK(), 2);
            Assert.assertEquals(param.getTopP(), 2f);
            Assert.assertEquals(param.getTfsZ(), 2f);
            Assert.assertEquals(param.getTypicalP(), 2f);
            Assert.assertEquals(param.getTemperature(), 2f);
            Assert.assertEquals(param.getRepeatPenalty(), 2f);
            Assert.assertEquals(param.getRepeatLastN(), 2);
            Assert.assertEquals(param.getFrequencyPenalty(), 2f);

            Assert.assertEquals(param.getFrequencyPenalty(), 2f);
            Assert.assertEquals(param.getPresencePenalty(), 2f);
            Assert.assertTrue(param.isPenalizeNl());
            Assert.assertTrue(param.isIgnoreEos());
            Assert.assertEquals(param.getMirostat(), 2);
            Assert.assertEquals(param.getMirostatTau(), 2f);
            Assert.assertEquals(param.getMirostatEta(), 2f);
            Assert.assertEquals(param.getNumberBeams(), 5);
            Assert.assertEquals(param.getSeed(), 2);
            Map<Integer, Float> logitBias = param.getLogitBias();
            Assert.assertNotNull(logitBias);
            Assert.assertEquals(logitBias.size(), 2);
            Assert.assertEquals(logitBias.get(2), 0.4f);
            Assert.assertNotNull(param.getGrammar());
            Assert.assertNotNull(param.getAntiPrompt()[0], "User: ");
        }
    }
}
