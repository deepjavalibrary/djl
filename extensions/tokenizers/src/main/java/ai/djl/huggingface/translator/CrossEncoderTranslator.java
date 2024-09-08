/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.PairList;
import ai.djl.util.StringPair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** The translator for Huggingface cross encoder model. */
public class CrossEncoderTranslator implements Translator<StringPair, float[]> {

    private HuggingFaceTokenizer tokenizer;
    private boolean includeTokenTypes;
    private boolean sigmoid;
    private Batchifier batchifier;

    CrossEncoderTranslator(
            HuggingFaceTokenizer tokenizer,
            boolean includeTokenTypes,
            boolean sigmoid,
            Batchifier batchifier) {
        this.tokenizer = tokenizer;
        this.includeTokenTypes = includeTokenTypes;
        this.sigmoid = sigmoid;
        this.batchifier = batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, StringPair input) {
        Encoding encoding = tokenizer.encode(input.getKey(), input.getValue());
        ctx.setAttachment("encoding", encoding);
        return encoding.toNDList(ctx.getNDManager(), includeTokenTypes);
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchProcessInput(TranslatorContext ctx, List<StringPair> inputs) {
        NDManager manager = ctx.getNDManager();
        PairList<String, String> list = new PairList<>(inputs);
        Encoding[] encodings = tokenizer.batchEncode(list);
        NDList[] batch = new NDList[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            batch[i] = encodings[i].toNDList(manager, includeTokenTypes);
        }
        return batchifier.batchify(batch);
    }

    /** {@inheritDoc} */
    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {
        NDArray logits = list.get(0);
        if (sigmoid) {
            logits = logits.getNDArrayInternal().sigmoid();
        }
        return logits.toFloatArray();
    }

    /** {@inheritDoc} */
    @Override
    public List<float[]> batchProcessOutput(TranslatorContext ctx, NDList list) {
        if (sigmoid) {
            NDList[] batches = batchifier.unbatchify(list);
            List<float[]> ret = new ArrayList<>(batches.length);
            for (NDList batch : batches) {
                NDArray result = batch.get(0);
                result = result.getNDArrayInternal().sigmoid();
                ret.add(result.toFloatArray());
            }
            return ret;
        }
        NDArray array = list.get(0);
        int batchSize = Math.toIntExact(array.size(0));
        float[] buf = list.get(0).toFloatArray();
        if (batchSize == 1) {
            return Collections.singletonList(buf);
        }

        int length = buf.length / batchSize;
        List<float[]> ret = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize; ++i) {
            float[] f = new float[length];
            System.arraycopy(buf, i * length, f, 0, length);
            ret.add(f);
        }
        return ret;
    }

    /**
     * Creates a builder to build a {@code CrossEncoderTranslator}.
     *
     * @param tokenizer the tokenizer
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer) {
        return new Builder(tokenizer);
    }

    /**
     * Creates a builder to build a {@code CrossEncoderTranslator}.
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

    /** The builder for question answering translator. */
    public static final class Builder {

        private HuggingFaceTokenizer tokenizer;
        private boolean includeTokenTypes;
        private boolean sigmoid = true;
        private Batchifier batchifier = Batchifier.STACK;

        Builder(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
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
         * Sets if apply sigmoid for the {@link Translator}.
         *
         * @param sigmoid true to apply sigmoid
         * @return this builder
         */
        public Builder optSigmoid(boolean sigmoid) {
            this.sigmoid = sigmoid;
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
            optIncludeTokenTypes(ArgumentsUtil.booleanValue(arguments, "includeTokenTypes"));
            optSigmoid(ArgumentsUtil.booleanValue(arguments, "sigmoid", true));
            String batchifierStr = ArgumentsUtil.stringValue(arguments, "batchifier", "stack");
            optBatchifier(Batchifier.fromString(batchifierStr));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException if I/O error occurs
         */
        public CrossEncoderTranslator build() throws IOException {
            return new CrossEncoderTranslator(tokenizer, includeTokenTypes, sigmoid, batchifier);
        }
    }
}
