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
package ai.djl.huggingface.tokenizers;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.StringReader;

/** The type Pad token resolver. */
public final class PadTokenResolver {

    private static final Logger logger = LoggerFactory.getLogger(PadTokenResolver.class);

    private PadTokenResolver() {
        throw new IllegalStateException("Utility class");
    }

    /**
     * Extracts pad token and ID from tokenizer.json and/or tokenizer_config.json. Follows HF
     * behavior: tokenizer.json takes precedence, config is fallback.
     *
     * @param tokenizerJson tokenizer.json content as string
     * @param config TokenizerConfig, may be null
     * @return PadInfo or null if pad token cannot be resolved
     */
    public static PadInfo extractPadInfo(String tokenizerJson, TokenizerConfig config) {
        try {
            JsonObject tokenizer =
                    JsonParser.parseReader(new StringReader(tokenizerJson)).getAsJsonObject();

            // 🔹 Priority 1: From tokenizer.json → padding block
            if (tokenizer.has("padding") && !tokenizer.get("padding").isJsonNull()) {
                JsonObject padding = tokenizer.getAsJsonObject("padding");
                if (padding != null && padding.has("pad_token") && padding.has("pad_id")) {
                    String padToken = padding.get("pad_token").getAsString();
                    int padId = padding.get("pad_id").getAsInt();
                    return new PadInfo(padToken, padId);
                }
            }

            // 🔹 Priority 2: From tokenizer_config.json (fallback)
            if (config != null && config.getPadToken() != null) {
                String padToken = config.getPadToken();

                // First: try to find it in added_tokens block (preferred over vocab)
                if (tokenizer.has("added_tokens")) {
                    JsonArray added = tokenizer.getAsJsonArray("added_tokens");
                    for (JsonElement el : added) {
                        JsonObject item = el.getAsJsonObject();
                        if (padToken.equals(item.get("content").getAsString())) {
                            int id = item.get("id").getAsInt();
                            return new PadInfo(padToken, id);
                        }
                    }
                }

                // Second: try to resolve from model.vocab (legacy scenario)
                if (tokenizer.has("model")) {
                    JsonObject model = tokenizer.getAsJsonObject("model");
                    if (model.has("vocab")) {
                        JsonObject vocab = model.getAsJsonObject("vocab");
                        JsonElement element = vocab.get(padToken);
                        if (element != null && element.isJsonPrimitive()) {
                            int id = element.getAsInt();
                            return new PadInfo(padToken, id);
                        }
                    }
                }

                // Could not resolve the ID but we return the token for awareness
                logger.warn(
                        "pad_token '{}' was found in config, but not in tokenizer.json added_tokens"
                                + " or vocab",
                        padToken);
            }

        } catch (Exception e) {
            logger.warn("Failed to parse pad_token from tokenizer.json", e);
        }

        return null;
    }

    /** The type Pad info. */
    public static class PadInfo {
        /** The Pad token. */
        private final String padToken;

        /** The Pad id. */
        private final int padId;

        /**
         * Instantiates a new Pad info.
         *
         * @param padToken the pad token
         * @param padId the pad id
         */
        public PadInfo(String padToken, int padId) {
            this.padToken = padToken;
            this.padId = padId;
        }

        /**
         * Gets pad token.
         *
         * @return the pad token
         */
        public String getPadToken() {
            return padToken;
        }

        /**
         * Gets pad id.
         *
         * @return the pad id
         */
        public int getPadId() {
            return padId;
        }
    }
}
