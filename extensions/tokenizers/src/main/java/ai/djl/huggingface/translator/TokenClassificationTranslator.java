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
import ai.djl.huggingface.tokenizers.jni.CharSpan;
import ai.djl.modality.nlp.translator.NamedEntity;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** The translator for Huggingface token classification model. */
public class TokenClassificationTranslator implements Translator<String, NamedEntity[]> {

    private HuggingFaceTokenizer tokenizer;
    private boolean includeTokenTypes;
    private boolean int32;
    private boolean softmax;
    private Batchifier batchifier;
    private PretrainedConfig config;

    TokenClassificationTranslator(
            HuggingFaceTokenizer tokenizer,
            boolean includeTokenTypes,
            boolean int32,
            boolean softmax,
            Batchifier batchifier) {
        this.tokenizer = tokenizer;
        this.includeTokenTypes = includeTokenTypes;
        this.int32 = int32;
        this.softmax = softmax;
        this.batchifier = batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        Path path = ctx.getModel().getModelPath();
        Path file = path.resolve("config.json");
        try (Reader reader = Files.newBufferedReader(file)) {
            config = JsonUtils.GSON.fromJson(reader, PretrainedConfig.class);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        Encoding encoding = tokenizer.encode(input);
        ctx.setAttachment("encoding", encoding);
        return encoding.toNDList(ctx.getNDManager(), includeTokenTypes, int32);
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchProcessInput(TranslatorContext ctx, List<String> inputs) {
        NDManager manager = ctx.getNDManager();
        Encoding[] encodings = tokenizer.batchEncode(inputs);
        ctx.setAttachment("encodings", encodings);
        NDList[] batch = new NDList[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            batch[i] = encodings[i].toNDList(manager, includeTokenTypes, int32);
        }
        return batchifier.batchify(batch);
    }

    /** {@inheritDoc} */
    @Override
    public NamedEntity[] processOutput(TranslatorContext ctx, NDList list) {
        Encoding encoding = (Encoding) ctx.getAttachment("encoding");
        return toNamedEntities(encoding, list);
    }

    /** {@inheritDoc} */
    @Override
    public List<NamedEntity[]> batchProcessOutput(TranslatorContext ctx, NDList list) {
        NDList[] batch = batchifier.unbatchify(list);
        Encoding[] encodings = (Encoding[]) ctx.getAttachment("encodings");
        List<NamedEntity[]> ret = new ArrayList<>(batch.length);
        for (int i = 0; i < batch.length; ++i) {
            ret.add(toNamedEntities(encodings[i], batch[i]));
        }
        return ret;
    }

    /**
     * Creates a builder to build a {@code TokenClassificationTranslator}.
     *
     * @param tokenizer the tokenizer
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer) {
        return new Builder(tokenizer);
    }

    /**
     * Creates a builder to build a {@code TokenClassificationTranslator}.
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

    private NamedEntity[] toNamedEntities(Encoding encoding, NDList list) {
        long[] inputIds = encoding.getIds();
        CharSpan[] offsetMapping = encoding.getCharTokenSpans();
        long[] specialTokenMasks = encoding.getSpecialTokenMask();
        NDArray probabilities = list.get(0);
        if (softmax) {
            probabilities = probabilities.softmax(1);
        }

        List<NamedEntity> entities = new ArrayList<>();

        for (int i = 0; i < inputIds.length; ++i) {
            if (specialTokenMasks[i] != 0) {
                continue;
            }

            int entityIdx = (int) probabilities.get(i).argMax().getLong();
            String entity = config.id2label.get(String.valueOf(entityIdx));

            if (!"O".equals(entity)) {
                float score = probabilities.get(i).getFloat(entityIdx);
                String word = encoding.getTokens()[i];
                int start = offsetMapping[i].getStart();
                int end = offsetMapping[i].getEnd();

                NamedEntity item = new NamedEntity(entity, score, i, word, start, end);
                entities.add(item);
            }
        }
        return entities.toArray(new NamedEntity[0]);
    }

    /** The builder for token classification translator. */
    public static final class Builder {

        private HuggingFaceTokenizer tokenizer;
        private boolean includeTokenTypes;
        private boolean int32;
        private boolean softmax = true;
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
         * Sets if implement softmax operation for the {@link Translator}.
         *
         * @param softmax true to implement softmax to model output result
         * @return this builder
         */
        public Builder optSoftmax(boolean softmax) {
            this.softmax = softmax;
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
            optInt32(ArgumentsUtil.booleanValue(arguments, "int32"));
            optSoftmax(ArgumentsUtil.booleanValue(arguments, "softmax", true));
            String batchifierStr = ArgumentsUtil.stringValue(arguments, "batchifier", "stack");
            optBatchifier(Batchifier.fromString(batchifierStr));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException if I/O error occurs
         */
        public TokenClassificationTranslator build() throws IOException {
            return new TokenClassificationTranslator(
                    tokenizer, includeTokenTypes, int32, softmax, batchifier);
        }
    }
}
