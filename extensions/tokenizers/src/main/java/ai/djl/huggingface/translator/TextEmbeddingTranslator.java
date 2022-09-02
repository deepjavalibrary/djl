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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.util.Map;

/** The translator for Huggingface text embedding model. */
public class TextEmbeddingTranslator implements Translator<String, float[]> {

    private static final int[] AXIS = {0};

    private HuggingFaceTokenizer tokenizer;
    private Batchifier batchifier;

    TextEmbeddingTranslator(HuggingFaceTokenizer tokenizer, Batchifier batchifier) {
        this.tokenizer = tokenizer;
        this.batchifier = batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        NDManager manager = ctx.getNDManager();
        Encoding encoding = tokenizer.encode(input);
        ctx.setAttachment("encoding", encoding);
        long[] indices = encoding.getIds();
        long[] attentionMask = encoding.getAttentionMask();
        NDList ndList = new NDList(2);
        ndList.add(manager.create(indices));
        ndList.add(manager.create(attentionMask));
        return ndList;
    }

    /** {@inheritDoc} */
    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {
        NDArray embeddings = list.get("last_hidden_state");
        Encoding encoding = (Encoding) ctx.getAttachment("encoding");
        long[] attentionMask = encoding.getAttentionMask();
        NDManager manager = ctx.getNDManager();
        NDArray inputAttentionMask = manager.create(attentionMask).toType(DataType.FLOAT32, true);
        long[] shape = embeddings.getShape().getShape();
        inputAttentionMask = inputAttentionMask.tile(shape[shape.length - 1]);
        inputAttentionMask = inputAttentionMask.reshape(embeddings.getShape());
        NDArray inputAttentionMaskSum = inputAttentionMask.sum(AXIS);
        NDArray clamp = inputAttentionMaskSum.clip(1e-9, 1e12);
        NDArray prod = embeddings.mul(inputAttentionMask);
        NDArray sum = prod.sum(AXIS);
        embeddings = sum.div(clamp).normalize(2, 0);

        return embeddings.toFloatArray();
    }

    /**
     * Creates a builder to build a {@code TextEmbeddingTranslator}.
     *
     * @param tokenizer the tokenizer
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer) {
        return new Builder(tokenizer);
    }

    /**
     * Creates a builder to build a {@code TextEmbeddingTranslator}.
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

    /** The builder for token classification translator. */
    public static final class Builder {

        private HuggingFaceTokenizer tokenizer;
        private Batchifier batchifier = Batchifier.STACK;

        Builder(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
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
            String batchifierStr = ArgumentsUtil.stringValue(arguments, "batchifier", "stack");
            optBatchifier(Batchifier.fromString(batchifierStr));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException if I/O error occurs
         */
        public TextEmbeddingTranslator build() throws IOException {
            return new TextEmbeddingTranslator(tokenizer, batchifier);
        }
    }
}
