/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.huggingface.tokenizers;

import ai.djl.util.JsonUtils;

import com.google.gson.JsonSyntaxException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.math.BigDecimal;

public class TokenizerConfigTest {

    private static final BigDecimal MODEL_MAX_LENGTH_THRESHOLD = new BigDecimal("1e20");

    @Test
    public void testModelMaxLength() {
        String[] configs = {"{}", "{\"model_max_length\":512}", "{\"model_max_length\":\" 512 \"}"};
        for (String json : configs) {
            TokenizerConfig config = JsonUtils.GSON.fromJson(json, TokenizerConfig.class);
            Assert.assertEquals(config.getModelMaxLength(), 512);
            Assert.assertFalse(config.isModelMaxLengthUnknown());
        }

        TokenizerConfig config =
                JsonUtils.GSON.fromJson("{\"model_max_length\":-1}", TokenizerConfig.class);
        Assert.assertEquals(config.getModelMaxLength(), -1);
        Assert.assertFalse(config.isModelMaxLengthUnknown());
    }

    @Test
    public void testUnknownModelMaxLength() {
        String aboveThreshold = MODEL_MAX_LENGTH_THRESHOLD.add(BigDecimal.ONE).toPlainString();
        String[] values = {
            "1000000000000000019884624838656", // Hugging Face's default for an unknown model limit.
            "1e30",
            aboveThreshold,
            aboveThreshold + ".0000"
        };
        for (String value : values) {
            String json = "{\"model_max_length\":" + value + "}";
            TokenizerConfig config = JsonUtils.GSON.fromJson(json, TokenizerConfig.class);
            Assert.assertEquals(config.getModelMaxLength(), -1);
            Assert.assertTrue(config.isModelMaxLengthUnknown());
        }
    }

    @Test
    public void testInvalidModelMaxLength() {
        BigDecimal fractionalLength = MODEL_MAX_LENGTH_THRESHOLD.add(new BigDecimal("1.5"));
        String[] values = {
            "512.5",
            Long.toString((long) Integer.MAX_VALUE + 1),
            MODEL_MAX_LENGTH_THRESHOLD.toPlainString(),
            fractionalLength.toPlainString()
        };
        for (String value : values) {
            String json = "{\"model_max_length\":" + value + "}";
            Assert.assertThrows(
                    JsonSyntaxException.class,
                    () -> JsonUtils.GSON.fromJson(json, TokenizerConfig.class));
        }
    }
}
