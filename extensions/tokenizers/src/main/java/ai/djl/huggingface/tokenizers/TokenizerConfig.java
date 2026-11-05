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

import ai.djl.util.JsonUtils;

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.annotations.JsonAdapter;
import com.google.gson.annotations.SerializedName;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.Type;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Configuration class for HuggingFace tokenizer. Loads and holds configuration from
 * tokenizer_config.json.
 */
public class TokenizerConfig {
    private static final Logger logger = LoggerFactory.getLogger(TokenizerConfig.class);

    // Transformers uses model_max_length > int(1e20) to indicate an unknown model limit.
    private static final BigDecimal UNKNOWN_LENGTH_THRESHOLD = new BigDecimal("1e20");

    /** The constant DEFAULT_MAX_LENGTH. */
    public static final int DEFAULT_MAX_LENGTH = 512;

    @SerializedName("tokenizer_class")
    private String tokenizerClass;

    // Retain the sentinel value to distinguish it from a literal -1.
    @SerializedName("model_max_length")
    @JsonAdapter(ModelMaxLengthDeserializer.class)
    private BigDecimal modelMaxLength;

    @SerializedName("strip_accents")
    private Boolean stripAccents;

    @SerializedName("clean_up_tokenization_spaces")
    private Boolean cleanUpTokenizationSpaces;

    @SerializedName("add_prefix_space")
    private Boolean addPrefixSpace;

    // Special tokens
    @SerializedName("bos_token")
    private String bosToken;

    @SerializedName("eos_token")
    private String eosToken;

    @SerializedName("unk_token")
    private String unkToken;

    @SerializedName("sep_token")
    private String sepToken;

    @SerializedName("pad_token")
    private String padToken;

    @SerializedName("cls_token")
    private String clsToken;

    /**
     * Load tokenizer config.
     *
     * @param configPath the config path
     * @return the tokenizer config
     */
    public static TokenizerConfig load(Path configPath) {
        if (Files.exists(configPath)) {
            try (Reader reader = Files.newBufferedReader(configPath)) {
                return JsonUtils.GSON.fromJson(reader, TokenizerConfig.class);
            } catch (IOException e) {
                logger.warn(
                        "Failed to load tokenizer_config.json, falling back to legacy config", e);
                return null;
            }
        }
        return null;
    }

    /**
     * Gets model max length.
     *
     * <p>Hugging Face uses integer values greater than {@code 1e20} when no model limit is known.
     * These values are returned as {@code -1}.
     *
     * @return the model max length, or -1 when the configuration marks it as unknown
     */
    public int getModelMaxLength() {
        if (Objects.isNull(modelMaxLength)) {
            return DEFAULT_MAX_LENGTH;
        }
        if (isModelMaxLengthUnknown()) {
            return -1;
        }
        return modelMaxLength.intValue();
    }

    /**
     * Returns whether the configuration marks the model max length as unknown.
     *
     * @return true if the configured length is a Hugging Face unknown-length sentinel
     */
    boolean isModelMaxLengthUnknown() {
        return modelMaxLength != null && modelMaxLength.compareTo(UNKNOWN_LENGTH_THRESHOLD) > 0;
    }

    /**
     * Is strip accents boolean.
     *
     * @return the boolean
     */
    public boolean isStripAccents() {
        return Boolean.TRUE.equals(stripAccents);
    }

    /**
     * Is clean up tokenization spaces boolean.
     *
     * @return the boolean
     */
    public boolean isCleanUpTokenizationSpaces() {
        return Boolean.TRUE.equals(cleanUpTokenizationSpaces);
    }

    /**
     * Is add prefix space boolean.
     *
     * @return the boolean
     */
    public boolean isAddPrefixSpace() {
        return Boolean.TRUE.equals(addPrefixSpace);
    }

    /**
     * Gets bos token.
     *
     * @return the bos token
     */
    public String getBosToken() {
        return bosToken;
    }

    /**
     * Gets eos token.
     *
     * @return the eos token
     */
    public String getEosToken() {
        return eosToken;
    }

    /**
     * Gets unk token.
     *
     * @return the unk token
     */
    public String getUnkToken() {
        return unkToken;
    }

    /**
     * Gets sep token.
     *
     * @return the sep token
     */
    public String getSepToken() {
        return sepToken;
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
     * Gets cls token.
     *
     * @return the cls token
     */
    public String getClsToken() {
        return clsToken;
    }

    /**
     * Gets tokenizer class.
     *
     * @return the tokenizer class
     */
    public String getTokenizerClass() {
        return tokenizerClass;
    }

    /**
     * Has explicit strip accents boolean.
     *
     * @return the boolean
     */
    public boolean hasExplicitStripAccents() {
        return stripAccents != null;
    }

    /**
     * Has explicit add prefix space boolean.
     *
     * @return the boolean
     */
    public boolean hasExplicitAddPrefixSpace() {
        return addPrefixSpace != null;
    }

    private static final class ModelMaxLengthDeserializer implements JsonDeserializer<BigDecimal> {

        /** {@inheritDoc} */
        @Override
        public BigDecimal deserialize(
                JsonElement json, Type type, JsonDeserializationContext context) {
            if (json.isJsonPrimitive()) {
                BigDecimal configuredModelMaxLength;
                try {
                    configuredModelMaxLength = json.getAsBigDecimal();
                } catch (NumberFormatException e) {
                    // Gson's Integer adapter accepts numeric formats that BigDecimal does not.
                    return parseIntegerModelMaxLength(json);
                }
                if (configuredModelMaxLength.compareTo(UNKNOWN_LENGTH_THRESHOLD) > 0) {
                    // Integer values have zero or negative scale after removing trailing zeros.
                    boolean isIntegerValue =
                            configuredModelMaxLength.stripTrailingZeros().scale() <= 0;
                    if (isIntegerValue) {
                        return configuredModelMaxLength;
                    }
                }
            }
            return parseIntegerModelMaxLength(json);
        }

        private static BigDecimal parseIntegerModelMaxLength(JsonElement json) {
            // Parse JSON text to preserve the existing Integer validation.
            // Reading JsonElement directly can silently truncate fractions or overflow.
            Integer validatedModelMaxLength =
                    JsonUtils.GSON.fromJson(json.toString(), Integer.class);
            return BigDecimal.valueOf(validatedModelMaxLength);
        }
    }
}
