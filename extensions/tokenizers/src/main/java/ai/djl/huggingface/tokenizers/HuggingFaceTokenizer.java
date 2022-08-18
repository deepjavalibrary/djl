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
    private static final long DEFAULT_TRUNCATION_LENGTH = 512L;
    private static final long DEFAULT_PADDING_LENGTH = 0L;
    private static final long DEFAULT_STRIDE = 0L;
    private static final long DEFAULT_PAD_TO_MULTIPLE_OF = 0L;

    private boolean addSpecialTokens;
    private TruncationStrategy truncationStrategy;
    private PaddingStrategy paddingStrategy;
    private long truncationLength;
    private long paddingLength;
    private long stride;
    private long padToMultipleOf;

    private HuggingFaceTokenizer(long handle, Map<String, String> options) {
        super(handle);
        this.addSpecialTokens =
                options == null
                        || !options.containsKey("addSpecialTokens")
                        || Boolean.parseBoolean(options.get("addSpecialTokens"));
        if (options != null) {
            this.truncationStrategy =
                    TruncationStrategy.fromValue(
                            options.getOrDefault("truncationStrategy", "do_not_truncate"));
            this.paddingStrategy =
                    PaddingStrategy.fromValue(
                            options.getOrDefault("paddingStrategy", "do_not_pad"));
            this.truncationLength =
                    options.containsKey("maxLength")
                            ? Long.parseLong(options.get("maxLength"))
                            : DEFAULT_TRUNCATION_LENGTH;
            this.paddingLength =
                    options.containsKey("length")
                            ? Long.parseLong(options.get("length"))
                            : DEFAULT_PADDING_LENGTH;
            this.stride =
                    options.containsKey("stride")
                            ? Long.parseLong(options.get("stride"))
                            : DEFAULT_STRIDE;
            this.padToMultipleOf =
                    options.containsKey("padToMultipleOf")
                            ? Long.parseLong(options.get("padToMultipleOf"))
                            : DEFAULT_PAD_TO_MULTIPLE_OF;
            resolvePotentialTruncationAndPaddingConflicts();
            setTruncation();
            setPadding();
        }
    }

    /**
     * Create a pre-trained {@code HuggingFaceTokenizer} instance from huggingface hub.
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
     * @param enableTruncation whether to enable default truncation behavior
     */
    public void enableTruncation(boolean enableTruncation) {
        TruncationStrategy strategy =
                enableTruncation
                        ? TruncationStrategy.LONGEST_FIRST
                        : TruncationStrategy.NO_TRUNCATION;
        enableTruncation(strategy, DEFAULT_TRUNCATION_LENGTH, DEFAULT_STRIDE);
    }

    /**
     * Enables specific truncation behavior for the tokenizer.
     *
     * @param truncationStrategy the {@link TruncationStrategy} to use
     * @param maxLength the maximum length sequences should be truncated to
     * @param stride the length of the previous first sequence to include in the overflowing
     *     sequence
     */
    public void enableTruncation(
            TruncationStrategy truncationStrategy, long maxLength, long stride) {
        if (this.truncationStrategy == truncationStrategy
                && this.truncationLength == maxLength
                && this.stride == stride) {
            return;
        }
        updateTruncationAndPaddingSettings(
                truncationStrategy,
                maxLength,
                stride,
                this.paddingStrategy,
                this.paddingLength,
                this.padToMultipleOf);
        resolvePotentialTruncationAndPaddingConflicts();
        setTruncation();
    }

    /**
     * Enables or Disables default padding behavior for the tokenizer.
     *
     * @param enablePadding whether to enable default padding behavior
     */
    public void enablePadding(boolean enablePadding) {
        PaddingStrategy strategy =
                enablePadding ? PaddingStrategy.LONGEST : PaddingStrategy.NO_PADDING;
        enablePadding(strategy, DEFAULT_PADDING_LENGTH, DEFAULT_PAD_TO_MULTIPLE_OF);
    }

    /**
     * Enables specific padding behavior for the tokenizer.
     *
     * @param paddingStrategy the {@link PaddingStrategy} to use
     * @param length if PaddingStrategy.MAX_LENGTH is used, determines the length sequences are
     *     padded to
     * @param padToMultipleOf if non-zero, pads sequences to a multiple of the value
     */
    public void enablePadding(PaddingStrategy paddingStrategy, long length, long padToMultipleOf) {
        if (this.paddingStrategy == paddingStrategy
                && this.paddingLength == length
                && this.padToMultipleOf == padToMultipleOf) {
            return;
        }
        updateTruncationAndPaddingSettings(
                this.truncationStrategy,
                this.truncationLength,
                this.stride,
                paddingStrategy,
                length,
                padToMultipleOf);
        resolvePotentialTruncationAndPaddingConflicts();
        setPadding();
    }

    private void resolvePotentialTruncationAndPaddingConflicts() {
        if (this.truncationStrategy == TruncationStrategy.NO_TRUNCATION
                && this.paddingStrategy == PaddingStrategy.NO_PADDING) {
            return;
        }
        if (TruncationStrategy.NO_TRUNCATION != this.truncationStrategy
                && this.truncationLength < this.stride) {
            throw new IllegalArgumentException("maxLength cannot be less than stride");
        }
        if (TruncationStrategy.NO_TRUNCATION != this.truncationStrategy
                && this.padToMultipleOf > 0
                && this.truncationLength % this.padToMultipleOf != 0) {
            throw new IllegalArgumentException(
                    "maxLength and padToMultipleOf are both specified but maxLength is not a"
                            + " multiple of padToMultipleOf");
        }
        if (PaddingStrategy.LONGEST == this.paddingStrategy && this.paddingLength != 0) {
            logger.warn(
                    "padding length has no effect when padding strategy \"longest\" is used. Use"
                            + " padding strategy \"max_length\" to pad to a specific length");
            this.paddingLength = 0;
        }
        if (PaddingStrategy.MAX_LENGTH == this.paddingStrategy && this.paddingLength == 0) {
            logger.warn(
                    "padding strategy of \"max_length\" requested with length of 0."
                            + " Defaulting to padding strategy \"longest\"");
            this.paddingStrategy = PaddingStrategy.LONGEST;
        }
    }

    private void setTruncation() {
        if (TruncationStrategy.NO_TRUNCATION == this.truncationStrategy) {
            TokenizersLibrary.LIB.disableTruncation(getHandle());
        } else {
            TokenizersLibrary.LIB.setTruncation(
                    getHandle(),
                    this.truncationLength,
                    this.truncationStrategy.getValue(),
                    this.stride);
        }
    }

    private void setPadding() {
        if (PaddingStrategy.NO_PADDING == this.paddingStrategy) {
            TokenizersLibrary.LIB.disablePadding(getHandle());
        } else {
            TokenizersLibrary.LIB.setPadding(
                    getHandle(),
                    this.paddingLength,
                    this.paddingStrategy.getValue(),
                    this.padToMultipleOf);
        }
    }

    private void updateTruncationAndPaddingSettings(
            TruncationStrategy truncationStrategy,
            long maxLength,
            long stride,
            PaddingStrategy paddingStrategy,
            long length,
            long padToMultipleOf) {
        this.truncationStrategy = truncationStrategy;
        this.truncationLength = maxLength;
        this.stride = stride;
        this.paddingStrategy = paddingStrategy;
        this.paddingLength = length;
        this.padToMultipleOf = padToMultipleOf;
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
    public enum TruncationStrategy {
        LONGEST_FIRST("longest_first"),
        ONLY_FIRST("only_first"),
        ONLY_SECOND("only_second"),
        NO_TRUNCATION("do_not_truncate");

        private String value;

        TruncationStrategy(String value) {
            this.value = value;
        }

        /**
         * Returns the String representation of the TruncationStrategy type.
         *
         * @return the String representation of the TruncationStrategy type
         */
        public String getValue() {
            return this.value;
        }

        /**
         * Converts the String to the matching TruncationStrategy type.
         *
         * @param value the String to convert
         * @return the matching TruncationStrategy type
         * @throws IllegalArgumentException if the value does not match any TruncationStrategy type
         */
        public static TruncationStrategy fromValue(String value) {
            for (TruncationStrategy strategy : TruncationStrategy.values()) {
                if (strategy.getValue().equals(value)) {
                    return strategy;
                }
            }
            throw new IllegalArgumentException(
                    String.format("The value [%s] does not match any TruncationStrategy", value));
        }
    }

    /** An enum to represent the different available padding strategies. */
    public enum PaddingStrategy {
        LONGEST("longest"),
        MAX_LENGTH("max_length"),
        NO_PADDING("do_not_pad");

        private String value;

        PaddingStrategy(String value) {
            this.value = value;
        }

        /**
         * Returns the String representation of the PaddingStrategy type.
         *
         * @return the String representation of the PaddingStrategy type
         */
        public String getValue() {
            return this.value;
        }

        /**
         * Converts the String to the matching PaddingStrategy type.
         *
         * @param value the String to convert
         * @return the matching PaddingStrategy type
         * @throws IllegalArgumentException if the value does not match any PaddingStrategy type
         */
        public static PaddingStrategy fromValue(String value) {
            for (PaddingStrategy strategy : PaddingStrategy.values()) {
                if (strategy.getValue().equals(value)) {
                    return strategy;
                }
            }
            throw new IllegalArgumentException(
                    String.format("The value [%s] does not match any PaddingStrategy", value));
        }
    }
}
