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
import ai.djl.ndarray.NDManager;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.util.Ec2Utils;
import ai.djl.util.NativeResource;
import ai.djl.util.PairList;
import ai.djl.util.Platform;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code HuggingFaceTokenizer} is a Huggingface tokenizer implementation of the {@link Tokenizer}
 * interface that converts sentences into token.
 */
public final class HuggingFaceTokenizer extends NativeResource<Long> implements Tokenizer {

    private static final Logger logger = LoggerFactory.getLogger(HuggingFaceTokenizer.class);

    private boolean addSpecialTokens;
    private boolean withOverflowingTokens;
    private Locale doLowerCase;
    private TruncationStrategy truncation;
    private PaddingStrategy padding;
    private int maxLength;
    private int stride;
    private int padToMultipleOf;
    private int modelMaxLength;

    private HuggingFaceTokenizer(long handle, Map<String, String> options) {
        super(handle);
        truncation = TruncationStrategy.LONGEST_FIRST;
        padding = PaddingStrategy.LONGEST;
        maxLength = TokenizersLibrary.LIB.getMaxLength(handle);
        stride = TokenizersLibrary.LIB.getStride(handle);
        padToMultipleOf = TokenizersLibrary.LIB.getPadToMultipleOf(handle);

        if (options != null) {
            String val = options.getOrDefault("addSpecialTokens", "true");
            addSpecialTokens = Boolean.parseBoolean(val);
            val = options.getOrDefault("withOverflowingTokens", "false");
            withOverflowingTokens = Boolean.parseBoolean(val);
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
            String lowerCase = options.getOrDefault("doLowerCase", "false");
            if ("true".equals(lowerCase)) {
                this.doLowerCase = Locale.getDefault();
            } else if (!"false".equals(lowerCase)) {
                this.doLowerCase = Locale.forLanguageTag(lowerCase);
            }
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
        Ec2Utils.callHome("Huggingface");
        LibUtils.checkStatus();
        String autoToken = Utils.getEnvOrSystemProperty("HF_TOKEN");
        if (options != null) {
            autoToken = options.getOrDefault("hf_token", autoToken);
        }

        long handle = TokenizersLibrary.LIB.createTokenizer(identifier, autoToken);
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
     * Create a pre-trained BPE {@code HuggingFaceTokenizer} instance from existing models.
     *
     * @param vocab the BPE vocabulary file
     * @param merges the BPE merges file
     * @param options tokenizer options
     * @return a {@code HuggingFaceTokenizer} instance
     * @throws IOException when IO operation fails in loading a resource
     */
    public static HuggingFaceTokenizer newInstance(
            Path vocab, Path merges, Map<String, String> options) throws IOException {
        Ec2Utils.callHome("Huggingface");
        LibUtils.checkStatus();

        String vocabFile = vocab.toAbsolutePath().toString();
        String mergesFile = merges.toAbsolutePath().toString();
        long handle = TokenizersLibrary.LIB.createBpeTokenizer(vocabFile, mergesFile);
        return new HuggingFaceTokenizer(handle, options);
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
        Ec2Utils.callHome("Huggingface");
        LibUtils.checkStatus();
        String json = Utils.toString(is);

        long handle = TokenizersLibrary.LIB.createTokenizerFromString(json);
        return new HuggingFaceTokenizer(handle, options);
    }

    /**
     * Returns the version of the Huggingface tokenizer.
     *
     * @return the version number of the Huggingface tokenizer
     */
    public String getVersion() {
        Platform platform = Platform.detectPlatform("tokenizers");
        return platform.getVersion();
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
     * @param withOverflowingTokens whether to return overflowing tokens
     * @return the {@code Encoding} of the input sentence
     */
    public Encoding encode(String text, boolean addSpecialTokens, boolean withOverflowingTokens) {
        if (text == null) {
            throw new NullPointerException("text cannot be null");
        }
        if (doLowerCase != null) {
            text = text.toLowerCase(doLowerCase);
        }
        long encoding = TokenizersLibrary.LIB.encode(getHandle(), text, addSpecialTokens);
        return toEncoding(encoding, withOverflowingTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentence.
     *
     * @param text the input sentence
     * @return the {@code Encoding} of the input sentence
     */
    public Encoding encode(String text) {
        return encode(text, addSpecialTokens, withOverflowingTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentence.
     *
     * @param text the input sentence
     * @param textPair the second input sentence
     * @param addSpecialTokens whether to encode the sequence with special tokens relative to their
     *     model
     * @param withOverflowingTokens whether to return overflowing tokens
     * @return the {@code Encoding} of the input sentence
     */
    public Encoding encode(
            String text, String textPair, boolean addSpecialTokens, boolean withOverflowingTokens) {
        if (text == null || textPair == null) {
            throw new NullPointerException("text/text_pair cannot be null");
        }

        if (doLowerCase != null) {
            text = text.toLowerCase(doLowerCase);
            textPair = textPair.toLowerCase(doLowerCase);
        }

        long encoding =
                TokenizersLibrary.LIB.encodeDual(getHandle(), text, textPair, addSpecialTokens);
        return toEncoding(encoding, withOverflowingTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentence.
     *
     * @param text the input sentence
     * @param textPair the second input sentence
     * @return the {@code Encoding} of the input sentence
     */
    public Encoding encode(String text, String textPair) {
        return encode(text, textPair, addSpecialTokens, withOverflowingTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentences.
     *
     * @param inputs the input sentences
     * @param addSpecialTokens whether to encode the sequence with special tokens relative to their
     *     model
     * @param withOverflowingTokens whether to return overflowing tokens
     * @return the {@code Encoding} of the input sentences
     */
    public Encoding encode(
            List<String> inputs, boolean addSpecialTokens, boolean withOverflowingTokens) {
        String[] array = inputs.toArray(Utils.EMPTY_ARRAY);
        return encode(array, addSpecialTokens, withOverflowingTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentences.
     *
     * @param inputs the input sentences
     * @return the {@code Encoding} of the input sentences
     */
    public Encoding encode(List<String> inputs) {
        return encode(inputs, addSpecialTokens, withOverflowingTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentences.
     *
     * @param inputs the input sentences
     * @param addSpecialTokens whether to encode the sequence with special tokens relative to their
     *     model
     * @param withOverflowingTokens whether to return overflowing tokens
     * @return the {@code Encoding} of the input sentences
     */
    public Encoding encode(
            String[] inputs, boolean addSpecialTokens, boolean withOverflowingTokens) {
        if (doLowerCase != null) {
            for (int i = 0; i < inputs.length; ++i) {
                inputs[i] = inputs[i].toLowerCase(doLowerCase);
            }
        } else if (Arrays.stream(inputs).anyMatch(Objects::isNull)) {
            throw new NullPointerException("input text cannot be null");
        }
        long encoding = TokenizersLibrary.LIB.encodeList(getHandle(), inputs, addSpecialTokens);
        return toEncoding(encoding, withOverflowingTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentences.
     *
     * @param inputs the input sentences
     * @return the {@code Encoding} of the input sentences
     */
    public Encoding encode(String[] inputs) {
        return encode(inputs, addSpecialTokens, withOverflowingTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentence in batch.
     *
     * @param inputs the batch of input sentence
     * @param addSpecialTokens whether to encode the sequence with special tokens relative to their
     *     model
     * @param withOverflowingTokens whether to return overflowing tokens
     * @return the {@code Encoding} of the input sentence in batch
     */
    public Encoding[] batchEncode(
            List<String> inputs, boolean addSpecialTokens, boolean withOverflowingTokens) {
        String[] array = inputs.toArray(Utils.EMPTY_ARRAY);
        return batchEncode(array, addSpecialTokens, withOverflowingTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentence in batch.
     *
     * @param inputs the batch of input sentence
     * @return the {@code Encoding} of the input sentence in batch
     */
    public Encoding[] batchEncode(List<String> inputs) {
        return batchEncode(inputs, addSpecialTokens, withOverflowingTokens);
    }

    /**
     * Returns the {@code Encoding} of the input sentence in batch.
     *
     * @param inputs the batch of input sentence
     * @param addSpecialTokens whether to encode the sequence with special tokens relative to their
     *     model
     * @param withOverflowingTokens whether to return overflowing tokens
     * @return the {@code Encoding} of the input sentence in batch
     */
    public Encoding[] batchEncode(
            String[] inputs, boolean addSpecialTokens, boolean withOverflowingTokens) {
        if (doLowerCase != null) {
            for (int i = 0; i < inputs.length; ++i) {
                inputs[i] = inputs[i].toLowerCase(doLowerCase);
            }
        } else if (Arrays.stream(inputs).anyMatch(Objects::isNull)) {
            throw new NullPointerException("input text cannot be null");
        }
        long[] encodings = TokenizersLibrary.LIB.batchEncode(getHandle(), inputs, addSpecialTokens);
        Encoding[] ret = new Encoding[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            ret[i] = toEncoding(encodings[i], withOverflowingTokens);
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
        return batchEncode(inputs, addSpecialTokens, withOverflowingTokens);
    }

    /**
     * Returns the {@code Encoding} of the input text pair in batch.
     *
     * @param inputs the batch of input text pair
     * @param addSpecialTokens whether to encode the sequence with special tokens relative to their
     *     model
     * @param withOverflowingTokens whether to return overflowing tokens
     * @return the {@code Encoding} of the input text pair in batch
     */
    public Encoding[] batchEncode(
            PairList<String, String> inputs,
            boolean addSpecialTokens,
            boolean withOverflowingTokens) {
        String[] text = inputs.keyArray(Utils.EMPTY_ARRAY);
        String[] textPair = inputs.valueArray(Utils.EMPTY_ARRAY);
        if (doLowerCase != null) {
            for (int i = 0; i < text.length; ++i) {
                text[i] = text[i].toLowerCase(doLowerCase);
            }
            for (int i = 0; i < textPair.length; ++i) {
                textPair[i] = textPair[i].toLowerCase(doLowerCase);
            }
        } else {
            if (inputs.keys().stream().anyMatch(Objects::isNull)) {
                throw new NullPointerException("text pair key cannot be null");
            }
            if (inputs.values().stream().anyMatch(Objects::isNull)) {
                throw new NullPointerException("text pair value cannot be null");
            }
        }
        long[] encodings =
                TokenizersLibrary.LIB.batchEncodePair(
                        getHandle(), text, textPair, addSpecialTokens);
        Encoding[] ret = new Encoding[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            ret[i] = toEncoding(encodings[i], withOverflowingTokens);
        }
        return ret;
    }

    /**
     * Returns the {@code Encoding} of the input text pair in batch.
     *
     * @param inputs the batch of input text pair
     * @return the {@code Encoding} of the input text pair in batch
     */
    public Encoding[] batchEncode(PairList<String, String> inputs) {
        return batchEncode(inputs, addSpecialTokens, withOverflowingTokens);
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
     * Returns the decoded Strings from the input batch ids.
     *
     * @param batchIds the batch of id sequences to decode
     * @param skipSpecialTokens whether to remove special tokens in the decoding
     * @return the decoded Strings from the input batch ids
     */
    public String[] batchDecode(long[][] batchIds, boolean skipSpecialTokens) {
        return TokenizersLibrary.LIB.batchDecode(getHandle(), batchIds, skipSpecialTokens);
    }

    /**
     * Returns the decoded Strings from the input batch ids.
     *
     * @param batchIds the batch of id sequences to decode
     * @return the decoded Strings from the input batch ids
     */
    public String[] batchDecode(long[][] batchIds) {
        return batchDecode(batchIds, !addSpecialTokens);
    }

    /**
     * Returns the truncation policy.
     *
     * @return the truncation policy
     */
    public String getTruncation() {
        return truncation.name();
    }

    /**
     * Returns the padding policy.
     *
     * @return the padding policy
     */
    public String getPadding() {
        return padding.name();
    }

    /**
     * Returns the max token length.
     *
     * @return the max token length
     */
    public int getMaxLength() {
        return maxLength;
    }

    /**
     * Returns the stride to use in overflow overlap when truncating sequences longer than the model
     * supports.
     *
     * @return the stride to use in overflow overlap when truncating sequences longer than the model
     *     supports
     */
    public int getStride() {
        return stride;
    }

    /**
     * Returns the padToMultipleOf for padding.
     *
     * @return the padToMultipleOf for padding
     */
    public int getPadToMultipleOf() {
        return padToMultipleOf;
    }

    /**
     * Creates a builder to build a {@code HuggingFaceTokenizer}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code HuggingFaceTokenizer}.
     *
     * @param arguments the models' arguments
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = builder();
        builder.configure(arguments);

        return builder;
    }

    /*
     * See: https://huggingface.co/docs/transformers/pad_truncation
     */
    private void updateTruncationAndPadding() {
        boolean isTruncate = truncation != TruncationStrategy.DO_NOT_TRUNCATE;
        if (padding == PaddingStrategy.MAX_LENGTH || isTruncate) {
            if (maxLength == -1) {
                logger.warn(
                        "maxLength is not explicitly specified, use modelMaxLength: {}",
                        modelMaxLength);
                maxLength = modelMaxLength;
            } else if (maxLength > modelMaxLength) {
                logger.warn(
                        "maxLength is greater then modelMaxLength, change to: {}", modelMaxLength);
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
                            "maxLength ({}) is not a multiple of padToMultipleOf ({}), change to:"
                                    + " {}",
                            maxLength,
                            padToMultipleOf,
                            newMaxLength);
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

    private Encoding toEncoding(long encoding, boolean withOverflowingTokens) {
        long[] ids = TokenizersLibrary.LIB.getTokenIds(encoding);
        long[] typeIds = TokenizersLibrary.LIB.getTypeIds(encoding);
        String[] tokens = TokenizersLibrary.LIB.getTokens(encoding);
        long[] wordIds = TokenizersLibrary.LIB.getWordIds(encoding);
        long[] attentionMask = TokenizersLibrary.LIB.getAttentionMask(encoding);
        long[] specialTokenMask = TokenizersLibrary.LIB.getSpecialTokenMask(encoding);
        CharSpan[] charSpans = TokenizersLibrary.LIB.getTokenCharSpans(encoding);

        int overFlowCount = TokenizersLibrary.LIB.getOverflowCount(encoding);
        boolean exceedMaxLength = overFlowCount > 0;
        Encoding[] overflowing;
        if (withOverflowingTokens) {
            long[] overflowingHandles = TokenizersLibrary.LIB.getOverflowing(encoding);
            overflowing = new Encoding[overflowingHandles.length];
            for (int i = 0; i < overflowingHandles.length; ++i) {
                overflowing[i] = toEncoding(overflowingHandles[i], true);
            }
        } else {
            overflowing = new Encoding[0];
        }

        TokenizersLibrary.LIB.deleteEncoding(encoding);
        return new Encoding(
                ids,
                typeIds,
                tokens,
                wordIds,
                attentionMask,
                specialTokenMask,
                charSpans,
                exceedMaxLength,
                overflowing);
    }

    /** {@inheritDoc} */
    @SuppressWarnings("deprecation")
    @Override
    protected void finalize() throws Throwable {
        close();
        super.finalize();
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

    /** The builder for creating huggingface tokenizer. */
    public static final class Builder {

        private NDManager manager;
        private Map<String, String> options;

        Builder() {
            options = new ConcurrentHashMap<>();
            options.put("addSpecialTokens", "true");
        }

        /**
         * Sets the optional manager used to manage the lifecycle of the tokenizer.
         *
         * @param manager the {@link NDManager}
         * @return this builder
         */
        public Builder optManager(NDManager manager) {
            this.manager = manager;
            return this;
        }

        /**
         * Sets the name of the tokenizer.
         *
         * @param tokenizerName the name of the tokenizer
         * @return this builder
         */
        public Builder optTokenizerName(String tokenizerName) {
            options.put("tokenizer", tokenizerName);
            return this;
        }

        /**
         * Sets the file path of the tokenizer.
         *
         * @param tokenizerPath the path of the tokenizer
         * @return this builder
         */
        public Builder optTokenizerPath(Path tokenizerPath) {
            options.putIfAbsent("tokenizerPath", tokenizerPath.toString());
            return this;
        }

        /**
         * Sets if add special tokens.
         *
         * @param addSpecialTokens true to add special tokens
         * @return this builder
         */
        public Builder optAddSpecialTokens(boolean addSpecialTokens) {
            options.put("addSpecialTokens", String.valueOf(addSpecialTokens));
            return this;
        }

        /**
         * Sets if add special tokens.
         *
         * @param withOverflowingTokens true to return overflowing tokens
         * @return this builder
         */
        public Builder optWithOverflowingTokens(boolean withOverflowingTokens) {
            options.put("withOverflowingTokens", String.valueOf(withOverflowingTokens));
            return this;
        }

        /**
         * Enables or Disables default truncation behavior for the tokenizer.
         *
         * @param enabled whether to enable default truncation behavior
         * @return this builder
         */
        public Builder optTruncation(boolean enabled) {
            options.put("truncation", String.valueOf(enabled));
            return this;
        }

        /**
         * Enables truncation to only truncate the first item.
         *
         * @return this builder
         */
        public Builder optTruncateFirstOnly() {
            options.put("truncation", TruncationStrategy.ONLY_FIRST.name());
            return this;
        }

        /**
         * Enables truncation to only truncate the second item.
         *
         * @return this builder
         */
        public Builder optTruncateSecondOnly() {
            options.put("truncation", TruncationStrategy.ONLY_SECOND.name());
            return this;
        }

        /**
         * Enables or Disables default padding behavior for the tokenizer.
         *
         * @param enabled whether to enable default padding behavior
         * @return this builder
         */
        public Builder optPadding(boolean enabled) {
            options.put("padding", String.valueOf(enabled));
            return this;
        }

        /**
         * Enables padding to pad sequences to previously specified maxLength, or modelMaxLength if
         * not specified.
         *
         * @return this builder
         */
        public Builder optPadToMaxLength() {
            options.put("padding", PaddingStrategy.MAX_LENGTH.name());
            return this;
        }

        /**
         * Sets maxLength for padding and truncation.
         *
         * @param maxLength the length to truncate and/or pad sequences to
         * @return this builder
         */
        public Builder optMaxLength(int maxLength) {
            options.put("maxLength", String.valueOf(maxLength));
            return this;
        }

        /**
         * Sets padToMultipleOf for padding.
         *
         * @param padToMultipleOf the multiple of sequences should be padded to
         * @return this builder
         */
        public Builder optPadToMultipleOf(int padToMultipleOf) {
            options.put("padToMultipleOf", String.valueOf(padToMultipleOf));
            return this;
        }

        /**
         * Sets the stride to use in overflow overlap when truncating sequences longer than the
         * model supports.
         *
         * @param stride the number of tokens to overlap when truncating long sequences
         * @return this builder
         */
        public Builder optStride(int stride) {
            options.put("stride", String.valueOf(stride));
            return this;
        }

        /**
         * Sets the doLowerCase for the tokenizer.
         *
         * @param doLowerCase {@code true} to enable convert to lowercase
         * @return this builder
         */
        public Builder optDoLowerCase(boolean doLowerCase) {
            options.put("doLowerCase", String.valueOf(doLowerCase));
            return this;
        }

        /**
         * Sets the doLowerCase for the tokenizer with specific locale.
         *
         * @param locale the locale to use when converting to lowercase
         * @return this builder
         */
        public Builder optDoLowerCase(String locale) {
            options.put("doLowerCase", locale);
            return this;
        }

        /**
         * Configures the builder with the arguments.
         *
         * @param arguments the arguments
         */
        public void configure(Map<String, ?> arguments) {
            for (Map.Entry<String, ?> entry : arguments.entrySet()) {
                options.put(entry.getKey(), entry.getValue().toString());
            }
        }

        /**
         * Utility to make a tokenizer managed by the builder manager (if one is specified).
         *
         * @param tokenizer the tokenizer to manage
         * @return the updated tokenizer
         */
        private HuggingFaceTokenizer managed(HuggingFaceTokenizer tokenizer) {
            if (manager != null) {
                manager.attachInternal(tokenizer.getUid(), tokenizer);
            }
            return tokenizer;
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException when IO operation fails in loading a resource
         */
        public HuggingFaceTokenizer build() throws IOException {
            String tokenizerName = options.get("tokenizer");
            if (tokenizerName != null) {
                return managed(HuggingFaceTokenizer.newInstance(tokenizerName, options));
            }
            String path = options.get("tokenizerPath");
            if (path == null) {
                throw new IllegalArgumentException("Missing tokenizer path.");
            }
            Path tokenizerPath = Paths.get(path);
            if (Files.isDirectory(tokenizerPath)) {
                Path tokenizerFile = tokenizerPath.resolve("tokenizer.json");
                if (Files.exists(tokenizerFile)) {
                    return managed(HuggingFaceTokenizer.newInstance(tokenizerPath, options));
                }
                Path vocab = tokenizerPath.resolve("vocab.json");
                Path merges = tokenizerPath.resolve("merges.txt");
                if (Files.exists(vocab) && Files.exists(merges)) {
                    return managed(HuggingFaceTokenizer.newInstance(vocab, merges, options));
                }
                throw new IOException("tokenizer.json file not found.");
            } else if (!Files.exists(tokenizerPath)) {
                throw new IOException("Tokenizer file not exits: " + tokenizerPath);
            }
            return managed(HuggingFaceTokenizer.newInstance(tokenizerPath, options));
        }
    }
}
