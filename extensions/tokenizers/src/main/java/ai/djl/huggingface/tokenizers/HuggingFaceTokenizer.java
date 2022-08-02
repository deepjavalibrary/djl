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

    private boolean addSpecialTokens;

    private HuggingFaceTokenizer(long handle, Map<String, String> options) {
        super(handle);
        this.addSpecialTokens =
                options == null
                        || !options.containsKey("addSpecialTokens")
                        || Boolean.parseBoolean(options.get("addSpecialTokens"));
    }

    /**
     * Create a pre-trained {@code HuggingFaceTokenizer} instance from hugginface hub.
     *
     * @param name the name of the huggingface tokenizer
     * @return a {@code HuggingFaceTokenizer} instance
     */
    public static HuggingFaceTokenizer newInstance(String name) {
        return newInstance(name, null);
    }

    /**
     * Create a pre-trained {@code HuggingFaceTokenizer} instance from hugginface hub.
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
     * Sets the truncation and padding behavior for the tokenizer.
     *
     * @param truncationStrategy the {@code TruncationStrategy} to use
     * @param paddingStrategy the {@code PaddingStrategy} to use
     * @param maxLength the maximum length to pad/truncate sequences to
     */
    public void setTruncationAndPadding(
            TruncationStrategy truncationStrategy,
            PaddingStrategy paddingStrategy,
            long maxLength) {
        setTruncationAndPadding(truncationStrategy, paddingStrategy, maxLength, 0, 0);
    }

    /**
     * Sets the truncation and padding behavior for the tokenizer.
     *
     * @param truncationStrategy the {@code TruncationStrategy} to use
     * @param paddingStrategy the {@code PaddingStrategy} to use
     * @param maxLength the maximum length to pad/truncate sequences to
     * @param stride value to use when handling overflow
     */
    public void setTruncationAndPadding(
            TruncationStrategy truncationStrategy,
            PaddingStrategy paddingStrategy,
            long maxLength,
            long stride) {
        setTruncationAndPadding(truncationStrategy, paddingStrategy, maxLength, stride, 0);
    }

    /**
     * Sets the truncation and padding behavior for the tokenizer.
     *
     * @param truncationStrategy the {@code TruncationStrategy} to use
     * @param paddingStrategy the {@code PaddingStrategy} to use
     * @param maxLength the maximum length to pad/truncate sequences to
     * @param stride value to use when handling overflow
     * @param padToMultipleOf pad sequence length to multiple of value
     */
    public void setTruncationAndPadding(
            TruncationStrategy truncationStrategy,
            PaddingStrategy paddingStrategy,
            long maxLength,
            long stride,
            long padToMultipleOf) {
        setTruncation(truncationStrategy, maxLength, stride);
        setPadding(paddingStrategy, maxLength, padToMultipleOf);
    }

    /**
     * Sets the truncation behavior for the tokenizer.
     *
     * @param truncationStrategy the {@code TruncationStrategy} to use
     * @param maxLength the maximum length to truncate sequences to
     * @param stride value to use when handling overflow
     */
    public void setTruncation(TruncationStrategy truncationStrategy, long maxLength, long stride) {
        if (truncationStrategy == TruncationStrategy.DO_NOT_TRUNCATE) {
            TokenizersLibrary.LIB.disableTruncation(getHandle());
        } else {
            TokenizersLibrary.LIB.setTruncation(
                    getHandle(), maxLength, truncationStrategy.toString().toLowerCase(), stride);
        }
    }

    /**
     * Sets the padding behavior for the tokenizer.
     *
     * @param paddingStrategy the {@code PaddingStrategy} to use
     * @param maxLength the maximum length to pad sequences to
     * @param padToMultipleOf pad sequence length to multiple of value
     */
    public void setPadding(PaddingStrategy paddingStrategy, long maxLength, long padToMultipleOf) {
        if (paddingStrategy == PaddingStrategy.DO_NOT_PAD) {
            TokenizersLibrary.LIB.disablePadding(getHandle());
        } else {
            TokenizersLibrary.LIB.setPadding(
                    getHandle(),
                    maxLength,
                    paddingStrategy.toString().toLowerCase(),
                    padToMultipleOf);
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

    enum TruncationStrategy {
        DO_NOT_TRUNCATE,
        LONGEST_FIRST,
        ONLY_FIRST,
        ONLY_SECOND
    }

    enum PaddingStrategy {
        DO_NOT_PAD,
        BATCH_LONGEST,
        FIXED
    }
}
