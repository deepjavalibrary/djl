/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.nlp.translator;

import ai.djl.modality.Input;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;

import com.google.gson.JsonParseException;
import com.google.gson.annotations.SerializedName;

/** A class that represents a {@code ZeroShotClassificationInput} object. */
public class ZeroShotClassificationInput {

    private String text;

    @SerializedName("candidate_labels")
    private String[] candidates;

    @SerializedName("multi_label")
    private boolean multiLabel;

    @SerializedName("hypothesis_template")
    private String hypothesisTemplate;

    /**
     * Constructs a new {@code ZeroShotClassificationInput} instance.
     *
     * @param text the text to classify
     * @param candidates the candidate labels
     */
    public ZeroShotClassificationInput(String text, String[] candidates) {
        this(text, candidates, false);
    }

    /**
     * Constructs a new {@code ZeroShotClassificationInput} instance.
     *
     * @param text the text to classify
     * @param candidates the candidate labels
     * @param multiLabel true to classify multiple labels
     */
    public ZeroShotClassificationInput(String text, String[] candidates, boolean multiLabel) {
        this(text, candidates, multiLabel, null);
    }

    /**
     * Constructs a new {@code ZeroShotClassificationInput} instance.
     *
     * @param text the text to classify
     * @param candidates the candidate labels
     * @param multiLabel true to classify multiple labels
     * @param hypothesisTemplate the custom template
     */
    public ZeroShotClassificationInput(
            String text, String[] candidates, boolean multiLabel, String hypothesisTemplate) {
        this.text = text;
        this.candidates = candidates;
        this.multiLabel = multiLabel;
        this.hypothesisTemplate = hypothesisTemplate;
    }

    /**
     * Returns the {@code ZeroShotClassificationInput} from the {@link Input}.
     *
     * @param input the input object
     * @return the {@code ZeroShotClassificationInput} from the {@link Input}
     * @throws TranslateException if the input is invalid
     */
    public static ZeroShotClassificationInput parseInput(Input input) throws TranslateException {
        String text = input.getData().getAsString();
        try {
            return JsonUtils.GSON.fromJson(text, ZeroShotClassificationInput.class);
        } catch (JsonParseException e) {
            throw new TranslateException("Input is not a valid json.", e);
        }
    }

    /**
     * Returns the text.
     *
     * @return the text to be classified
     */
    public String getText() {
        return text;
    }

    /**
     * Returns the candidate labels.
     *
     * @return the candidate labels
     */
    public String[] getCandidates() {
        return candidates;
    }

    /**
     * Returns {@code true} if to classify multiple labels.
     *
     * @return {@code true} if to classify multiple labels
     */
    public boolean isMultiLabel() {
        return multiLabel;
    }

    /**
     * Returns the custom template.
     *
     * @return the custom template
     */
    public String getHypothesisTemplate() {
        return hypothesisTemplate == null ? "This example is {}." : hypothesisTemplate;
    }
}
