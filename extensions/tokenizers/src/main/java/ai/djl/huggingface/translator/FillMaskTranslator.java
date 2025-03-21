/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.huggingface.translator;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/** The translator for Huggingface fill mask model. */
public class FillMaskTranslator implements Translator<String, Classifications> {

    private HuggingFaceTokenizer tokenizer;
    private String maskToken;
    private long maskTokenId;
    private int topK;
    private boolean includeTokenTypes;
    private boolean int32;
    private Batchifier batchifier;

    FillMaskTranslator(
            HuggingFaceTokenizer tokenizer,
            String maskToken,
            int topK,
            boolean includeTokenTypes,
            boolean int32,
            Batchifier batchifier) {
        this.tokenizer = tokenizer;
        this.maskToken = maskToken;
        this.topK = topK;
        this.includeTokenTypes = includeTokenTypes;
        this.int32 = int32;
        this.batchifier = batchifier;
        Encoding encoding = tokenizer.encode(maskToken, false, false);
        maskTokenId = encoding.getIds()[0];
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String input) throws TranslateException {
        Encoding encoding = tokenizer.encode(input);
        long[] indices = encoding.getIds();
        int maskIndex = getMaskIndex(indices);
        ctx.setAttachment("maskIndex", maskIndex);
        return encoding.toNDList(ctx.getNDManager(), includeTokenTypes, int32);
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchProcessInput(TranslatorContext ctx, List<String> inputs)
            throws TranslateException {
        NDManager manager = ctx.getNDManager();
        Encoding[] encodings = tokenizer.batchEncode(inputs);
        NDList[] batch = new NDList[encodings.length];
        int[] maskIndices = new int[encodings.length];
        ctx.setAttachment("maskIndices", maskIndices);
        for (int i = 0; i < batch.length; ++i) {
            long[] indices = encodings[i].getIds();
            maskIndices[i] = getMaskIndex(indices);
            batch[i] = encodings[i].toNDList(manager, includeTokenTypes, int32);
        }
        return batchifier.batchify(batch);
    }

    /** {@inheritDoc} */
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        int maskIndex = (int) ctx.getAttachment("maskIndex");
        return toClassifications(list, maskIndex);
    }

    /** {@inheritDoc} */
    @Override
    public List<Classifications> batchProcessOutput(TranslatorContext ctx, NDList list) {
        NDList[] batch = batchifier.unbatchify(list);
        int[] maskIndices = (int[]) ctx.getAttachment("maskIndices");
        List<Classifications> ret = new ArrayList<>(maskIndices.length);
        for (int i = 0; i < batch.length; ++i) {
            ret.add(toClassifications(batch[i], maskIndices[i]));
        }
        return ret;
    }

    private int getMaskIndex(long[] indices) throws TranslateException {
        int maskIndex = -1;
        for (int i = 0; i < indices.length; ++i) {
            if (indices[i] == maskTokenId) {
                if (maskIndex != -1) {
                    throw new TranslateException("Only one mask supported.");
                }
                maskIndex = i;
            }
        }
        if (maskIndex == -1) {
            throw new TranslateException("Mask token " + maskToken + " not found.");
        }
        return maskIndex;
    }

    private Classifications toClassifications(NDList output, int maskIndex) {
        NDArray prob = output.get(0).get(maskIndex).softmax(0);
        NDArray array = prob.argSort(0, false);
        long[] classIds = new long[topK];
        List<Double> probabilities = new ArrayList<>(topK);
        for (int i = 0; i < topK; ++i) {
            classIds[i] = array.getLong(i);
            probabilities.add((double) prob.getFloat(classIds[i]));
        }
        String[] classes = tokenizer.decode(classIds).trim().split(" ");
        return new Classifications(Arrays.asList(classes), probabilities);
    }

    /**
     * Creates a builder to build a {@code FillMaskTranslator}.
     *
     * @param tokenizer the tokenizer
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer) {
        return new Builder(tokenizer);
    }

    /**
     * Creates a builder to build a {@code FillMaskTranslator}.
     *
     * @param tokenizer the tokenizer
     * @param arguments the models' arguments
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer, Map<String, ?> arguments) {
        Builder builder = builder(tokenizer);
        builder.configure(arguments);

        return builder;
    }

    /** The builder for fill mask translator. */
    public static final class Builder {

        private HuggingFaceTokenizer tokenizer;
        private String maskedToken = "[MASK]";
        private int topK = 5;
        private boolean includeTokenTypes;
        private boolean int32;
        private Batchifier batchifier = Batchifier.STACK;

        Builder(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        /**
         * Sets the id of the mask {@link Translator}.
         *
         * @param maskedToken the id of the mask
         * @return this builder
         */
        public Builder optMaskToken(String maskedToken) {
            this.maskedToken = maskedToken;
            return this;
        }

        /**
         * Set the topK number of classes to be displayed.
         *
         * @param topK the number of top classes to return
         * @return this builder
         */
        public Builder optTopK(int topK) {
            this.topK = topK;
            return this;
        }

        /**
         * Sets if include token types for the {@link Translator}.
         *
         * @param includeTokenTypes true to include token types
         * @return this builder
         */
        public Builder optIncludeTokenTypes(boolean includeTokenTypes) {
            this.includeTokenTypes = includeTokenTypes;
            return this;
        }

        /**
         * Sets if use int32 datatype for the {@link Translator}.
         *
         * @param int32 true to include token types
         * @return this builder
         */
        public Builder optInt32(boolean int32) {
            this.int32 = int32;
            return this;
        }

        /**
         * Sets the {@link Batchifier} for the {@link Translator}.
         *
         * @param batchifier true to include token types
         * @return this builder
         */
        public Builder optBatchifier(Batchifier batchifier) {
            this.batchifier = batchifier;
            return this;
        }

        /**
         * Configures the builder with the model arguments.
         *
         * @param arguments the model arguments
         */
        public void configure(Map<String, ?> arguments) {
            optMaskToken(ArgumentsUtil.stringValue(arguments, "maskToken", "[MASK]"));
            optInt32(ArgumentsUtil.booleanValue(arguments, "int32"));
            optTopK(ArgumentsUtil.intValue(arguments, "topK", 5));
            optIncludeTokenTypes(ArgumentsUtil.booleanValue(arguments, "includeTokenTypes"));
            String batchifierStr = ArgumentsUtil.stringValue(arguments, "batchifier", "stack");
            optBatchifier(Batchifier.fromString(batchifierStr));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public FillMaskTranslator build() {
            return new FillMaskTranslator(
                    tokenizer, maskedToken, topK, includeTokenTypes, int32, batchifier);
        }
    }
}
