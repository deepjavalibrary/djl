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
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** The translator for Huggingface text embedding model. */
public class TextEmbeddingTranslator implements Translator<String, float[]> {

    private static final int[] AXIS = {-2};

    private HuggingFaceTokenizer tokenizer;
    private Batchifier batchifier;
    private boolean normalize;
    private String pooling;
    private boolean includeTokenTypes;
    private boolean int32;
    private String dense;
    private String denseActivation;
    private String layerNorm;
    private NDList denseModel;
    private NDList layerNormModel;

    TextEmbeddingTranslator(
            HuggingFaceTokenizer tokenizer,
            Batchifier batchifier,
            String pooling,
            boolean normalize,
            boolean includeTokenTypes,
            boolean int32,
            String dense,
            String denseActivation,
            String layerNorm) {
        this.tokenizer = tokenizer;
        this.batchifier = batchifier;
        this.pooling = pooling;
        this.normalize = normalize;
        this.includeTokenTypes = includeTokenTypes;
        this.int32 = int32;
        this.dense = dense;
        this.denseActivation = denseActivation;
        this.layerNorm = layerNorm;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        NDManager manager = ctx.getPredictorManager().newSubManager();
        if (dense != null) {
            Path file = Paths.get(dense);
            if (!file.isAbsolute()) {
                file = ctx.getModel().getModelPath().resolve(file);
            }
            if (Files.exists(file)) {
                try (InputStream is = Files.newInputStream(file)) {
                    denseModel = NDList.decode(manager, is);
                }
            }
        }
        if (layerNorm != null) {
            Path file = Paths.get(layerNorm);
            if (!file.isAbsolute()) {
                file = ctx.getModel().getModelPath().resolve(file);
            }
            if (Files.exists(file)) {
                try (InputStream is = Files.newInputStream(file)) {
                    layerNormModel = NDList.decode(manager, is);
                }
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        NDManager manager = ctx.getNDManager();
        Encoding encoding = tokenizer.encode(input);
        NDList list = encoding.toNDList(manager, includeTokenTypes, int32);
        ctx.setAttachment("attentionMask", list.get(1));
        return list;
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchProcessInput(TranslatorContext ctx, List<String> inputs) {
        NDManager manager = ctx.getNDManager();
        Encoding[] encodings = tokenizer.batchEncode(inputs);
        NDList list = Encoding.toNDList(encodings, manager, includeTokenTypes, int32);
        ctx.setAttachment("attentionMask", list.get(1));
        return list;
    }

    /** {@inheritDoc} */
    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {
        NDArray inputAttentionMask = (NDArray) ctx.getAttachment("attentionMask");
        NDArray embeddings = processEmbedding(list, inputAttentionMask);
        return embeddings.toFloatArray();
    }

    /** {@inheritDoc} */
    @Override
    public List<float[]> batchProcessOutput(TranslatorContext ctx, NDList list) {
        NDArray attentionMask = (NDArray) ctx.getAttachment("attentionMask");
        NDArray output = processEmbedding(list, attentionMask);
        int batchSize = Math.toIntExact(output.size(0));
        float[] buf = output.toFloatArray();
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

    private NDArray processEmbedding(NDList list, NDArray attentionMask) {
        NDArray embedding = list.get("last_hidden_state");
        if (embedding == null) {
            // For Onnx model, NDArray name is not present
            embedding = list.head();
        }
        switch (pooling) {
            case "mean":
                embedding = meanPool(embedding, attentionMask, false);
                break;
            case "mean_sqrt_len":
                embedding = meanPool(embedding, attentionMask, true);
                break;
            case "max":
                embedding = maxPool(embedding, attentionMask);
                break;
            case "weightedmean":
                embedding = weightedMeanPool(embedding, attentionMask);
                break;
            case "cls":
                embedding = embedding.get(new NDIndex(":, 0"));
                break;
            default:
                throw new AssertionError("Unexpected pooling mode: " + pooling);
        }

        if (denseModel != null) {
            NDArray weight = denseModel.get("linear.weight");
            NDArray bias = denseModel.get("linear.bias");
            embedding = embedding.getNDArrayInternal().linear(embedding, weight, bias).get(0);
            if ("Tanh".equalsIgnoreCase(denseActivation)) {
                embedding = embedding.tanh();
            }
        }
        if (layerNormModel != null) {
            NDArray weight = layerNormModel.get("norm.weight");
            NDArray bias = layerNormModel.get("norm.bias");
            Shape shape = weight.getShape();
            embedding =
                    embedding
                            .getNDArrayInternal()
                            .layerNorm(embedding, shape, weight, bias, 1e-5f)
                            .get(0);
        }
        if (normalize) {
            embedding = embedding.normalize(2, -1);
        }
        return embedding;
    }

    private static NDArray meanPool(NDArray embeddings, NDArray attentionMask, boolean sqrt) {
        long[] shape = embeddings.getShape().getShape();
        attentionMask = attentionMask.expandDims(-1).broadcast(shape);
        NDArray inputAttentionMaskSum = attentionMask.sum(AXIS);
        NDArray clamp = inputAttentionMaskSum.clip(1e-9f, 1e12f);
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

        return embeddings.max(AXIS, false);
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
        private String pooling = "mean";
        private boolean includeTokenTypes;
        private boolean int32;
        private String dense;
        private String denseActivation;
        private String layerNorm;

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
         * Sets the {@code normalize} for the {@link Translator}.
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
            if (!"mean".equals(poolingMode)
                    && !"max".equals(poolingMode)
                    && !"cls".equals(poolingMode)
                    && !"mean_sqrt_len".equals(poolingMode)
                    && !"weightedmean".equals(poolingMode)) {
                throw new IllegalArgumentException(
                        "Invalid pooling model, must be one of [mean, max, cls, mean_sqrt_len,"
                                + " weightedmean].");
            }
            this.pooling = poolingMode;
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
         * Sets the dense layer model file for the {@link Translator}.
         *
         * @param dense path to dense layer model file
         * @return this builder
         */
        public Builder optDense(String dense) {
            this.dense = dense;
            return this;
        }

        /**
         * Sets the dense activation function for the {@link Translator}.
         *
         * @param denseActivation path to dense layer
         * @return this builder
         */
        public Builder optDenseActivation(String denseActivation) {
            this.denseActivation = denseActivation;
            return this;
        }

        /**
         * Sets the LayerNorm model for the {@link Translator}.
         *
         * @param layerNorm path to LayerNorm model
         * @return this builder
         */
        public Builder optLayerNorm(String layerNorm) {
            this.layerNorm = layerNorm;
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
            optPoolingMode(ArgumentsUtil.stringValue(arguments, "pooling", "mean"));
            optIncludeTokenTypes(ArgumentsUtil.booleanValue(arguments, "includeTokenTypes"));
            optInt32(ArgumentsUtil.booleanValue(arguments, "int32"));
            optDense(ArgumentsUtil.stringValue(arguments, "dense"));
            optDenseActivation(ArgumentsUtil.stringValue(arguments, "denseActivation"));
            optLayerNorm(ArgumentsUtil.stringValue(arguments, "layerNorm"));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException if I/O error occurs
         */
        public TextEmbeddingTranslator build() throws IOException {
            return new TextEmbeddingTranslator(
                    tokenizer,
                    batchifier,
                    pooling,
                    normalize,
                    includeTokenTypes,
                    int32,
                    dense,
                    denseActivation,
                    layerNorm);
        }
    }
}
