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

import com.google.gson.annotations.SerializedName;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Configuration class for HuggingFace tokenizer. Loads and holds configuration from
 * tokenizer_config.json.
 */
public class TokenizerConfig {
    private static final Logger logger = LoggerFactory.getLogger(TokenizerConfig.class);

    /** The constant DEFAULT_MAX_LENGTH. */
    public static final int DEFAULT_MAX_LENGTH = 512;

    @SerializedName("tokenizer_class")
    private String tokenizerClass;

    @SerializedName("model_max_length")
    private Integer modelMaxLength;

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
     * @return the model max length
     */
    public int getModelMaxLength() {
        if (Objects.isNull(modelMaxLength)) {
            return DEFAULT_MAX_LENGTH;
        }
        return modelMaxLength;
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
}
