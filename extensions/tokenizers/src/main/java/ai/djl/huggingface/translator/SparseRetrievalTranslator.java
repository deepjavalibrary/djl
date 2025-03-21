/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.modality.nlp.EmbeddingOutput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Activation;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/** The translator handles sparse retrieval for Huggingface text embedding model. */
public class SparseRetrievalTranslator implements Translator<String, EmbeddingOutput> {

    private static final String[] SPECIAL_TOKENS = {
        "cls_token", "eos_token", "pad_token", "unk_token"
    };

    private HuggingFaceTokenizer tokenizer;
    private TextEmbeddingTranslator translator;
    private boolean includeTokenTypes;
    private boolean int32;
    private boolean returnDenseEmbedding;
    private Set<Long> unusedTokens;
    private String sparseLinear;
    private NDList sparseLinearModel;

    SparseRetrievalTranslator(Builder builder) {
        this.tokenizer = builder.tokenizer;
        this.translator = builder.baseBuilder.build();
        this.includeTokenTypes = builder.baseBuilder.includeTokenTypes;
        this.int32 = builder.baseBuilder.int32;
        this.returnDenseEmbedding = builder.returnDenseEmbedding;
        this.sparseLinear = builder.sparseLinear;
        Encoding encoding = tokenizer.encode(SPECIAL_TOKENS);
        unusedTokens = Arrays.stream(encoding.getIds()).boxed().collect(Collectors.toSet());
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        NDManager manager = ctx.getPredictorManager().newSubManager();
        if (returnDenseEmbedding) {
            translator.prepare(ctx);
        }
        if (sparseLinear != null) {
            Path file = Paths.get(sparseLinear);
            if (!file.isAbsolute()) {
                file = ctx.getModel().getModelPath().resolve(file);
            }
            if (Files.notExists(file)) {
                throw new TranslateException("sparseLinear file does not exist: " + sparseLinear);
            }
            try (InputStream is = Files.newInputStream(file)) {
                sparseLinearModel = NDList.decode(manager, is);
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        return batchProcessInput(ctx, Collections.singletonList(input));
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchProcessInput(TranslatorContext ctx, List<String> inputs) {
        NDManager manager = ctx.getNDManager();
        Encoding[] encodings = tokenizer.batchEncode(inputs);
        NDList list = Encoding.toNDList(encodings, manager, includeTokenTypes, int32);
        ctx.setAttachment("encodings", encodings);
        ctx.setAttachment("attentionMask", list.get(1));
        return list;
    }

    /** {@inheritDoc} */
    @Override
    public EmbeddingOutput processOutput(TranslatorContext ctx, NDList list) {
        return Objects.requireNonNull(batchProcessOutput(ctx, list)).get(0);
    }

    /** {@inheritDoc} */
    @Override
    public List<EmbeddingOutput> batchProcessOutput(TranslatorContext ctx, NDList list) {
        Encoding[] encodings = (Encoding[]) ctx.getAttachment("encodings");
        int batchSize = encodings.length;

        List<EmbeddingOutput> embeddings = new ArrayList<>();
        NDArray lastHiddenState = list.get("last_hidden_state");
        if (lastHiddenState == null) {
            lastHiddenState = list.get(0); // only pytorch returns "last_hidden_state" name
        }
        NDArray weight =
                sparseLinearModel.get("weight").toType(lastHiddenState.getDataType(), false);
        NDArray bias = sparseLinearModel.get("bias").toType(lastHiddenState.getDataType(), false);
        NDArray array =
                lastHiddenState.getNDArrayInternal().linear(lastHiddenState, weight, bias).get(0);
        array = Activation.relu(array);
        NDArray sparseVecs = array.squeeze(-1);
        float[] data = sparseVecs.toFloatArray();
        int index = 0;
        for (Encoding encoding : encodings) {
            long[] tokenIds = encoding.getIds();
            EmbeddingOutput embedding = new EmbeddingOutput();
            embeddings.add(embedding);
            for (long idx : tokenIds) {
                float w = data[index++];
                if (!unusedTokens.contains(idx) && w > 0) {
                    embedding.addTokenWeights(String.valueOf(idx), w);
                }
            }
        }

        if (returnDenseEmbedding) {
            NDArray attentionMask = (NDArray) ctx.getAttachment("attentionMask");
            NDArray output = translator.processEmbedding(list, attentionMask);
            FloatBuffer fb = output.toByteBuffer().asFloatBuffer();
            int denseEmbeddingSize = fb.remaining() / batchSize;
            for (EmbeddingOutput embedding : embeddings) {
                float[] buf = new float[denseEmbeddingSize];
                fb.get(buf);
                embedding.setDenseEmbedding(buf);
            }
        }
        return embeddings;
    }

    /**
     * Creates a builder to build a {@code SparseRetrievalTranslator}.
     *
     * @param tokenizer the tokenizer
     * @param arguments the models' arguments
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer, Map<String, ?> arguments) {
        Builder builder = new Builder(tokenizer);
        builder.configure(arguments);

        return builder;
    }

    /** The builder for question answering translator. */
    public static final class Builder {

        HuggingFaceTokenizer tokenizer;
        TextEmbeddingTranslator.Builder baseBuilder;
        boolean returnDenseEmbedding;
        String sparseLinear;

        Builder(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
            baseBuilder = TextEmbeddingTranslator.builder(tokenizer);
            sparseLinear = "sparse_linear.safetensors";
        }

        /**
         * Sets if apply sigmoid for the {@link Translator}.
         *
         * @param returnDenseEmbedding true to output dense embedding
         * @return this builder
         */
        public Builder optReturnDenseEmbedding(boolean returnDenseEmbedding) {
            this.returnDenseEmbedding = returnDenseEmbedding;
            return this;
        }

        /**
         * Sets the sparse linear layer model file for the {@link Translator}.
         *
         * @param sparseLinear path to sparse linear layer model file
         * @return this builder
         */
        public Builder optSparseLinear(String sparseLinear) {
            this.sparseLinear = sparseLinear;
            return this;
        }

        /**
         * Configures the builder with the model arguments.
         *
         * @param arguments the model arguments
         */
        public void configure(Map<String, ?> arguments) {
            baseBuilder.configure(arguments);
            optReturnDenseEmbedding(
                    ArgumentsUtil.booleanValue(arguments, "returnDenseEmbedding", false));
            optSparseLinear(ArgumentsUtil.stringValue(arguments, "sparseLinear", sparseLinear));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException if I/O error occurs
         */
        public SparseRetrievalTranslator build() throws IOException {
            return new SparseRetrievalTranslator(this);
        }
    }
}
