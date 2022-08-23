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

import ai.djl.huggingface.tokenizers.jni.CharSpan;
import ai.djl.huggingface.tokenizers.jni.LibUtils;
import ai.djl.huggingface.tokenizers.jni.TokenizersLibrary;
import ai.djl.modality.nlp.preprocess.Tokenizer;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.util.NativeResource;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * {@code HuggingFaceTokenizer} is a Huggingface tokenizer implementation of the {@link Tokenizer}
 * interface that converts sentences into token.
 */
public final class HuggingFaceTokenizer extends NativeResource<Long> implements Tokenizer {

    private static final Logger logger = LoggerFactory.getLogger(HuggingFaceTokenizer.class);

    private boolean addSpecialTokens;
    private TruncationStrategy truncation;
    private PaddingStrategy padding;
    private int maxLength;
    private int stride;
    private int padToMultipleOf;
    private int modelMaxLength;

    private HuggingFaceTokenizer(long handle, Map<String, String> options) {
        super(handle);
        String val = TokenizersLibrary.LIB.getTruncationStrategy(handle);
        truncation = TruncationStrategy.fromValue(val);
        val = TokenizersLibrary.LIB.getPaddingStrategy(handle);
        padding = PaddingStrategy.fromValue(val);
        maxLength = TokenizersLibrary.LIB.getMaxLength(handle);
        stride = TokenizersLibrary.LIB.getStride(handle);
        padToMultipleOf = TokenizersLibrary.LIB.getPadToMultipleOf(handle);

        if (options != null) {
            val = options.getOrDefault("addSpecialTokens", "true");
            addSpecialTokens = Boolean.parseBoolean(val);
            modelMaxLength = ArgumentsUtil.intValue(options, "modelMaxLength", 512);
            if (options.containsKey("truncation")) {
                truncation = TruncationStrategy.fromValue(options.get("truncation"));
            }
            if (options.containsKey("padding")) {
                padding = PaddingStrategy.fromValue(options.get("padding"));
            }
            maxLength = ArgumentsUtil.intValue(options, "maxLength", maxLength);
            stride = ArgumentsUtil.intValue(options, "stride", stride);
            padToMultipleOf = ArgumentsUtil.intValue(options, "padToMultipleOf", padToMultipleOf);
        } else {
            addSpecialTokens = true;
            modelMaxLength = 512;
        }

        updateTruncationAndPadding();
    }

    /**
     * Creates a pre-trained {@code HuggingFaceTokenizer} instance from huggingface hub.
     *
     * @param name the name of the huggingface tokenizer
     * @return a {@code HuggingFaceTokenizer} instance
     */
    public static HuggingFaceTokenizer newInstance(String name) {
        return newInstance(name, null);
    }

    /**
     * Create a pre-trained {@code HuggingFaceTokenizer} instance from huggingface hub.
     *
     * @param identifier the identifier of the huggingface tokenizer
     * @param options tokenizer options
     * @return a {@code HuggingFaceTokenizer} instance
     */
    public static HuggingFaceTokenizer newInstance(String identifier, Map<String, String> options) {
        LibUtils.checkStatus();

        long handle = TokenizersLibrary.LIB.createTokenizer(identifier);
        return new HuggingFaceTokenizer(handle, options);
    }

    /**
     * Create a pre-trained {@code HuggingFaceTokenizer} instance from existing models.
     *
     * @param modelPath the directory or file path of the model location
     * @return a {@code HuggingFaceTokenizer} instance
     * @throws IOException when IO operation fails in loading a resource
     */
    public static HuggingFaceTokenizer newInstance(Path modelPath) throws IOException {
        return newInstance(modelPath, null);
    }

    /**
     * Create a pre-trained {@code HuggingFaceTokenizer} instance from existing models.
     *
     * @param modelPath the directory or file path of the model location
     * @param options tokenizer options
     * @return a {@code HuggingFaceTokenizer} instance
     * @throws IOException when IO operation fails in loading a resource
     */
    public static HuggingFaceTokenizer newInstance(Path modelPath, Map<String, String> options)
            throws IOException {
        if (Files.isDirectory(modelPath)) {
            modelPath = modelPath.resolve("tokenizer.json");
        }
        try (InputStream is = Files.newInputStream(modelPath)) {
            return newInstance(is, options);
        }
    }

    /**
     * Create a pre-trained {@code HuggingFaceTokenizer} instance from {@code InputStream}.
     *
     * @param is {@code InputStream}
     * @param options tokenizer options
     * @return a {@code HuggingFaceTokenizer} instance
     * @throws IOException when IO operation fails in loading a resource
     */
    public static HuggingFaceTokenizer newInstance(InputStream is, Map<String, String> options)
            throws IOException {
        LibUtils.checkStatus();
        String json = Utils.toString(is);

        long handle = TokenizersLibrary.LIB.createTokenizerFromString(json);
        return new HuggingFaceTokenizer(handle, options);
    }

    /** {@inheritDoc} */
    @Override
    public List<String> tokenize(String sentence) {
        Encoding encoding = encode(sentence);
        return Arrays.asList(encoding.getTokens());
    }

    /** {@inheritDoc} */
    @Override
    public String buildSentence(List<String> tokens) {
        // TODO:
        return String.join(" ", tokens).replace(" ##", "").trim();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            TokenizersLibrary.LIB.deleteTokenizer(pointer);
        }
    }

    /**
     * Returns the {@code Encoding} of the input sentence.
     *
     * @param text the input sentence
     * @param addSpecialTokens whether to encode the sequence with special tokens relative to their
     *     model
     * @return the {@code Encoding} of the input sentence
     */
    public Encoding encode(String text, boolean addSpecialTokens) {
        long encoding = TokenizersLibrary.LIB.encode(getHandle(), text, addSpecialTokens);
        return toEncoding(encoding);
    }

    /**
     * Returns the {@code Encoding} of the input sentence.
     *
     * @param text the input sentence
     * @return the {@code Encoding} of the input sentence
     */
    public Encoding encode(String text) {
        return encode(text, addSpecialTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentence.
     *
     * @param text the input sentence
     * @param textPair the second input sentence
     * @param addSpecialTokens whether to encode the sequence with special tokens relative to their
     *     model
     * @return the {@code Encoding} of the input sentence
     */
    public Encoding encode(String text, String textPair, boolean addSpecialTokens) {
        long encoding =
                TokenizersLibrary.LIB.encodeDual(getHandle(), text, textPair, addSpecialTokens);
        return toEncoding(encoding);
    }

    /**
     * Returns the {@code Encoding} of the input sentence.
     *
     * @param text the input sentence
     * @param textPair the second input sentence
     * @return the {@code Encoding} of the input sentence
     */
    public Encoding encode(String text, String textPair) {
        return encode(text, textPair, addSpecialTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentences.
     *
     * @param inputs the input sentences
     * @param addSpecialTokens whether to encode the sequence with special tokens relative to their
     *     model
     * @return the {@code Encoding} of the input sentences
     */
    public Encoding encode(List<String> inputs, boolean addSpecialTokens) {
        String[] array = inputs.toArray(new String[0]);
        return encode(array, addSpecialTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentences.
     *
     * @param inputs the input sentences
     * @return the {@code Encoding} of the input sentences
     */
    public Encoding encode(List<String> inputs) {
        return encode(inputs, addSpecialTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentences.
     *
     * @param inputs the input sentences
     * @param addSpecialTokens whether to encode the sequence with special tokens relative to their
     *     model
     * @return the {@code Encoding} of the input sentences
     */
    public Encoding encode(String[] inputs, boolean addSpecialTokens) {
        long encoding = TokenizersLibrary.LIB.encodeList(getHandle(), inputs, addSpecialTokens);
        return toEncoding(encoding);
    }

    /**
     * Returns the {@code Encoding} of the input sentences.
     *
     * @param inputs the input sentences
     * @return the {@code Encoding} of the input sentences
     */
    public Encoding encode(String[] inputs) {
        return encode(inputs, addSpecialTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentence in batch.
     *
     * @param inputs the batch of input sentence
     * @param addSpecialTokens whether to encode the sequence with special tokens relative to their
     *     model
     * @return the {@code Encoding} of the input sentence in batch
     */
    public Encoding[] batchEncode(List<String> inputs, boolean addSpecialTokens) {
        String[] array = inputs.toArray(new String[0]);
        return batchEncode(array, addSpecialTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentence in batch.
     *
     * @param inputs the batch of input sentence
     * @return the {@code Encoding} of the input sentence in batch
     */
    public Encoding[] batchEncode(List<String> inputs) {
        return batchEncode(inputs, addSpecialTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentence in batch.
     *
     * @param inputs the batch of input sentence
     * @param addSpecialTokens whether to encode the sequence with special tokens relative to their
     *     model
     * @return the {@code Encoding} of the input sentence in batch
     */
    public Encoding[] batchEncode(String[] inputs, boolean addSpecialTokens) {
        long[] encodings = TokenizersLibrary.LIB.batchEncode(getHandle(), inputs, addSpecialTokens);
        Encoding[] ret = new Encoding[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            ret[i] = toEncoding(encodings[i]);
        }
        return ret;
    }

    /**
     * Returns the {@code Encoding} of the input sentence in batch.
     *
     * @param inputs the batch of input sentence
     * @return the {@code Encoding} of the input sentence in batch
     */
    public Encoding[] batchEncode(String[] inputs) {
        return batchEncode(inputs, addSpecialTokens);
    }

    /**
     * Returns the decoded String from the input ids.
     *
     * @param ids the input ids
     * @param skipSpecialTokens whether to remove special tokens in the decoding
     * @return the decoded String from the input ids
     */
    public String decode(long[] ids, boolean skipSpecialTokens) {
        return TokenizersLibrary.LIB.decode(getHandle(), ids, skipSpecialTokens);
    }

    /**
     * Returns the decoded String from the input ids.
     *
     * @param ids the input ids
     * @return the decoded String from the input ids
     */
    public String decode(long[] ids) {
        return decode(ids, !addSpecialTokens);
    }

    /**
     * Enables or Disables default truncation behavior for the tokenizer.
     *
     * @param enabled whether to enable default truncation behavior
     */
    public void setTruncation(boolean enabled) {
        TruncationStrategy strategy =
                enabled ? TruncationStrategy.LONGEST_FIRST : TruncationStrategy.DO_NOT_TRUNCATE;
        if (strategy != truncation) {
            truncation = strategy;
            updateTruncationAndPadding();
        }
    }

    /** Enables truncation to only truncate the first item. */
    public void setTruncateFirstOnly() {
        if (truncation != TruncationStrategy.ONLY_FIRST) {
            truncation = TruncationStrategy.ONLY_FIRST;
            updateTruncationAndPadding();
        }
    }

    /** Enables truncation to only truncate the second item. */
    public void setTruncateSecondOnly() {
        if (truncation != TruncationStrategy.ONLY_SECOND) {
            truncation = TruncationStrategy.ONLY_SECOND;
            updateTruncationAndPadding();
        }
    }

    /**
     * Enables or Disables default padding behavior for the tokenizer.
     *
     * @param enabled whether to enable default padding behavior
     */
    public void setPadding(boolean enabled) {
        PaddingStrategy p = enabled ? PaddingStrategy.LONGEST : PaddingStrategy.DO_NOT_PAD;
        if (padding != p) {
            padding = p;
            updateTruncationAndPadding();
        }
    }

    /**
     * Enables padding to pad sequences to previously specified maxLength, or modelMaxLength if not
     * specified.
     */
    public void setPaddingMaxLength() {
        if (padding != PaddingStrategy.MAX_LENGTH) {
            padding = PaddingStrategy.MAX_LENGTH;
            updateTruncationAndPadding();
        }
    }

    /**
     * Sets maxLength for padding and truncation.
     *
     * @param maxLength the length to truncate and/or pad sequences to
     */
    public void setMaxLength(int maxLength) {
        if (this.maxLength != maxLength) {
            this.maxLength = maxLength;
            updateTruncationAndPadding();
        }
    }

    /**
     * Sets padToMultipleOf for padding.
     *
     * @param padToMultipleOf the multiple of sequences should be padded to
     */
    public void setPadToMultipleOf(int padToMultipleOf) {
        if (this.padToMultipleOf != padToMultipleOf) {
            this.padToMultipleOf = padToMultipleOf;
            updateTruncationAndPadding();
        }
    }

    /*
     * See: https://huggingface.co/docs/transformers/pad_truncation
     */
    private void updateTruncationAndPadding() {
        boolean isTruncate = truncation != TruncationStrategy.DO_NOT_TRUNCATE;
        if (padding == PaddingStrategy.MAX_LENGTH || isTruncate) {
            if (maxLength == -1) {
                logger.warn(
                        "maxLength is not explicitly specified, use modelMaxLength: "
                                + modelMaxLength);
                maxLength = modelMaxLength;
            } else if (maxLength > modelMaxLength) {
                logger.warn(
                        "maxLength is greater then modelMaxLength, change to: " + modelMaxLength);
                maxLength = modelMaxLength;
            }

            if (padding == PaddingStrategy.MAX_LENGTH && isTruncate && padToMultipleOf != 0) {
                int remainder = maxLength % padToMultipleOf;
                if (remainder != 0) {
                    int newMaxLength = maxLength + padToMultipleOf - maxLength % padToMultipleOf;
                    if (newMaxLength > modelMaxLength) {
                        newMaxLength -= padToMultipleOf;
                    }
                    logger.warn(
                            "maxLength ("
                                    + maxLength
                                    + ") is not a multiple of padToMultipleOf ("
                                    + padToMultipleOf
                                    + "), change to: "
                                    + newMaxLength);
                    maxLength = newMaxLength;
                }
            }
        }

        if (isTruncate) {
            TokenizersLibrary.LIB.setTruncation(getHandle(), maxLength, truncation.name(), stride);
        } else {
            TokenizersLibrary.LIB.disableTruncation(getHandle());
        }

        if (padding == PaddingStrategy.DO_NOT_PAD) {
            TokenizersLibrary.LIB.disablePadding(getHandle());
        } else {
            TokenizersLibrary.LIB.setPadding(
                    getHandle(), maxLength, padding.name(), padToMultipleOf);
        }
    }

    private Encoding toEncoding(long encoding) {
        long[] ids = TokenizersLibrary.LIB.getTokenIds(encoding);
        long[] typeIds = TokenizersLibrary.LIB.getTypeIds(encoding);
        String[] tokens = TokenizersLibrary.LIB.getTokens(encoding);
        long[] wordIds = TokenizersLibrary.LIB.getWordIds(encoding);
        long[] attentionMask = TokenizersLibrary.LIB.getAttentionMask(encoding);
        long[] specialTokenMask = TokenizersLibrary.LIB.getSpecialTokenMask(encoding);
        CharSpan[] charSpans = TokenizersLibrary.LIB.getTokenCharSpans(encoding);

        TokenizersLibrary.LIB.deleteEncoding(encoding);
        return new Encoding(
                ids, typeIds, tokens, wordIds, attentionMask, specialTokenMask, charSpans);
    }

    /** An enum to represent the different available truncation strategies. */
    private enum TruncationStrategy {
        LONGEST_FIRST,
        ONLY_FIRST,
        ONLY_SECOND,
        DO_NOT_TRUNCATE;

        /**
         * Converts the String to the matching TruncationStrategy type.
         *
         * @param value the String to convert
         * @return the matching PaddingStrategy type
         * @throws IllegalArgumentException if the value does not match any TruncationStrategy type
         */
        static TruncationStrategy fromValue(String value) {
            if ("true".equals(value)) {
                return TruncationStrategy.LONGEST_FIRST;
            } else if ("false".equals(value)) {
                return TruncationStrategy.DO_NOT_TRUNCATE;
            }
            for (TruncationStrategy strategy : TruncationStrategy.values()) {
                if (strategy.name().equalsIgnoreCase(value)) {
                    return strategy;
                }
            }
            throw new IllegalArgumentException("Invalid TruncationStrategy: " + value);
        }
    }

    /** An enum to represent the different available padding strategies. */
    private enum PaddingStrategy {
        LONGEST,
        MAX_LENGTH,
        DO_NOT_PAD;

        /**
         * Converts the String to the matching PaddingStrategy type.
         *
         * @param value the String to convert
         * @return the matching PaddingStrategy type
         * @throws IllegalArgumentException if the value does not match any PaddingStrategy type
         */
        static PaddingStrategy fromValue(String value) {
            if ("true".equals(value)) {
                return PaddingStrategy.LONGEST;
            } else if ("false".equals(value)) {
                return PaddingStrategy.DO_NOT_PAD;
            }
            for (PaddingStrategy strategy : PaddingStrategy.values()) {
                if (strategy.name().equalsIgnoreCase(value)) {
                    return strategy;
                }
            }
            throw new IllegalArgumentException("Invalid PaddingStrategy: " + value);
        }
    }
}
