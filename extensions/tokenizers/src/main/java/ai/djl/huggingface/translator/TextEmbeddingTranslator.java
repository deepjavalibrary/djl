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
    private boolean normalize;
    private String pooling;

    TextEmbeddingTranslator(
            HuggingFaceTokenizer tokenizer,
            Batchifier batchifier,
            String pooling,
            boolean normalize) {
        this.tokenizer = tokenizer;
        this.batchifier = batchifier;
        this.pooling = pooling;
        this.normalize = normalize;
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
        NDManager manager = ctx.getNDManager();
        embeddings = processEmbedding(manager, embeddings, encoding, pooling);
        if (normalize) {
            embeddings = embeddings.normalize(2, 0);
        }

        return embeddings.toFloatArray();
    }

    /** {@inheritDoc} */
    @Override
    public TextEmbeddingBatchTranslator toBatchTranslator(Batchifier batchifier) {
        return new TextEmbeddingBatchTranslator(tokenizer, batchifier, pooling, normalize);
    }

    static NDArray processEmbedding(
            NDManager manager, NDArray embeddings, Encoding encoding, String pooling) {
        long[] attentionMask = encoding.getAttentionMask();
        NDArray inputAttentionMask = manager.create(attentionMask).toType(DataType.FLOAT32, true);
        switch (pooling) {
            case "mean_tokens":
                return meanPool(embeddings, inputAttentionMask, false);
            case "mean_sqrt_len_tokens":
                return meanPool(embeddings, inputAttentionMask, true);
            case "max_tokens":
                return maxPool(embeddings, inputAttentionMask);
            case "weightedmean_tokens":
                return weightedMeanPool(embeddings, inputAttentionMask);
            case "cls_token":
                return embeddings.get(0);
            default:
                throw new AssertionError("Unexpected pooling model: " + pooling);
        }
    }

    private static NDArray meanPool(NDArray embeddings, NDArray attentionMask, boolean sqrt) {
        long[] shape = embeddings.getShape().getShape();
        attentionMask = attentionMask.expandDims(-1).broadcast(shape);
        NDArray inputAttentionMaskSum = attentionMask.sum(AXIS);
        NDArray clamp = inputAttentionMaskSum.clip(1e-9, 1e12);
        NDArray prod = embeddings.mul(attentionMask);
        NDArray sum = prod.sum(AXIS);
        if (sqrt) {
            return sum.div(clamp.sqrt());
        }
        return sum.div(clamp);
    }

    private static NDArray maxPool(NDArray embeddings, NDArray inputAttentionMask) {
        long[] shape = embeddings.getShape().getShape();
        inputAttentionMask = inputAttentionMask.expandDims(-1).broadcast(shape);
        inputAttentionMask = inputAttentionMask.eq(0);
        embeddings = embeddings.duplicate();
        embeddings.set(inputAttentionMask, -1e9); // Set padding tokens to large negative value

        return embeddings.max(AXIS, true);
    }

    private static NDArray weightedMeanPool(NDArray embeddings, NDArray attentionMask) {
        long[] shape = embeddings.getShape().getShape();
        NDArray weight = embeddings.getManager().arange(1, shape[0] + 1);
        weight = weight.expandDims(-1).broadcast(shape);

        attentionMask = attentionMask.expandDims(-1).broadcast(shape).mul(weight);
        NDArray maskSum = attentionMask.sum(AXIS);
        NDArray embeddingSum = embeddings.mul(attentionMask).sum(AXIS);
        return embeddingSum.div(maskSum);
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
        private boolean normalize = true;
        private String pooling = "mean_tokens";

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
         * Sets the normalize for the {@link Translator}.
         *
         * @param normalize true to normalize the embeddings
         * @return this builder
         */
        public Builder optNormalize(boolean normalize) {
            this.normalize = normalize;
            return this;
        }

        /**
         * Sets the pooling for the {@link Translator}.
         *
         * @param poolingMode the pooling model, one of mean_pool, max_pool and cls
         * @return this builder
         */
        public Builder optPoolingMode(String poolingMode) {
            if (!"mean_tokens".equals(poolingMode)
                    && !"max_tokens".equals(poolingMode)
                    && !"cls_token".equals(poolingMode)
                    && !"mean_sqrt_len_tokens".equals(poolingMode)
                    && !"weightedmean_tokens".equals(poolingMode)) {
                throw new IllegalArgumentException(
                        "Invalid pooling model, must be one of [mean_tokens, max_tokens,"
                                + " cls_token, mean_sqrt_len_tokens, weightedmean_tokens].");
            }
            this.pooling = poolingMode;
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
            optNormalize(ArgumentsUtil.booleanValue(arguments, "normalize", true));
            optPoolingMode(ArgumentsUtil.stringValue(arguments, "pooling", "mean_tokens"));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException if I/O error occurs
         */
        public TextEmbeddingTranslator build() throws IOException {
            return new TextEmbeddingTranslator(tokenizer, batchifier, pooling, normalize);
        }
    }
}
