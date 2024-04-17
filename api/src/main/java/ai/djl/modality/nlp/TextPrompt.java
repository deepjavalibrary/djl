/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.nlp;

import ai.djl.modality.Input;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;

import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;

import java.util.List;

/** The input container for NLP text prompt. */
public final class TextPrompt {

    private String text;
    private List<String> batch;

    private TextPrompt(String text) {
        this.text = text;
    }

    private TextPrompt(List<String> batch) {
        this.batch = batch;
    }

    /**
     * Returns if the prompt is a batch.
     *
     * @return {@code true} if the prompt is a batch
     */
    public boolean isBatch() {
        return batch != null;
    }

    /**
     * Returns the single prompt.
     *
     * @return the single prompt
     */
    public String getText() {
        return text;
    }

    /**
     * Returns the batch prompt.
     *
     * @return the batch prompt
     */
    public List<String> getBatch() {
        return batch;
    }

    /**
     * Returns the {@code TextPrompt} from the {@link Input}.
     *
     * @param input the input object
     * @return the {@code TextPrompt} from the {@link Input}
     * @throws TranslateException if the input is invalid
     */
    public static TextPrompt parseInput(Input input) throws TranslateException {
        String contentType = input.getProperty("Content-Type", null);
        String text = input.getData().getAsString();
        if (!"application/json".equals(contentType)) {
            return new TextPrompt(text);
        }

        try {
            JsonElement element = JsonUtils.GSON.fromJson(text, JsonElement.class);
            if (element != null && element.isJsonObject()) {
                element = element.getAsJsonObject().get("inputs");
            }
            if (element == null) {
                throw new TranslateException("Missing \"inputs\" in json.");
            } else if (element.isJsonArray()) {
                List<String> batch = JsonUtils.GSON.fromJson(element, JsonUtils.LIST_TYPE);
                return new TextPrompt(batch);
            } else {
                return new TextPrompt(element.getAsString());
            }
        } catch (JsonParseException e) {
            throw new TranslateException("Input is not a valid json.", e);
        }
    }
}
