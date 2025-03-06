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
package ai.djl.modality.cv;

import ai.djl.modality.Input;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.annotations.SerializedName;

import java.io.IOException;

/** The input container for a vision language model. */
public class VisionLanguageInput {

    private Image image;
    private String text;

    @SerializedName("candidate_labels")
    private String[] candidates;

    @SerializedName("hypothesis_template")
    private String hypothesisTemplate;

    /**
     * Constructs a new {@code ImageTextInput} instance.
     *
     * @param image the image input
     * @param text the prompt
     */
    public VisionLanguageInput(Image image, String text) {
        this(image, text, null, null);
    }

    /**
     * Constructs a new {@code ImageTextInput} instance.
     *
     * @param image the image input
     * @param candidates the candidate labels
     */
    public VisionLanguageInput(Image image, String[] candidates) {
        this(image, null, candidates, null);
    }

    /**
     * Constructs a new {@code ImageTextInput} instance.
     *
     * @param image the image input
     * @param text the prompt
     * @param candidates the candidate labels
     * @param hypothesisTemplate the hypothesis template
     */
    public VisionLanguageInput(
            Image image, String text, String[] candidates, String hypothesisTemplate) {
        this.image = image;
        this.text = text;
        this.candidates = candidates;
        this.hypothesisTemplate = hypothesisTemplate;
    }

    /**
     * Returns the {@code ImageTextInput} from the {@link Input}.
     *
     * @param input the input object
     * @return the {@code ImageTextInput} from the {@link Input}
     * @throws TranslateException if the input is invalid
     * @throws IOException if failed to load image
     */
    public static VisionLanguageInput parseInput(Input input)
            throws TranslateException, IOException {
        String data = input.getData().getAsString();
        try {
            JsonObject obj = JsonUtils.GSON.fromJson(data, JsonObject.class);
            JsonElement url = obj.get("image");
            if (url == null) {
                url = obj.get("image_url");
            }
            if (url == null) {
                throw new TranslateException("Missing \"image\" parameter in input.");
            }
            Image img = ImageFactory.getInstance().fromUrl(url.getAsString());
            String text = JsonUtils.GSON.fromJson(obj.get("text"), String.class);
            String[] candidates =
                    JsonUtils.GSON.fromJson(obj.get("candidate_labels"), String[].class);
            String hypothesisTemplate =
                    JsonUtils.GSON.fromJson(obj.get("hypothesis_template"), String.class);
            return new VisionLanguageInput(img, text, candidates, hypothesisTemplate);
        } catch (JsonParseException e) {
            throw new TranslateException("Input is not a valid json.", e);
        }
    }

    /**
     * Returns the image input.
     *
     * @return the image input
     */
    public Image getImage() {
        return image;
    }

    /**
     * Sets the image input.
     *
     * @param image the image input
     */
    public void setImage(Image image) {
        this.image = image;
    }

    /**
     * Returns the prompt text.
     *
     * @return the prompt text
     */
    public String getText() {
        return text;
    }

    /**
     * Sets the prompt text.
     *
     * @param text the prompt text
     */
    public void setText(String text) {
        this.text = text;
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
     * Sets the candidate labels.
     *
     * @param candidates the candidate labels
     */
    public void setCandidates(String[] candidates) {
        this.candidates = candidates;
    }

    /**
     * Returns the hypothesis template.
     *
     * @return the hypothesis template
     */
    public String getHypothesisTemplate() {
        return hypothesisTemplate == null ? "This is a photo of {}." : hypothesisTemplate;
    }

    /**
     * Sets the hypothesis template.
     *
     * @param hypothesisTemplate the hypothesis template
     */
    public void setHypothesisTemplate(String hypothesisTemplate) {
        this.hypothesisTemplate = hypothesisTemplate;
    }
}
