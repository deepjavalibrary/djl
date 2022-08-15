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
import java.util.concurrent.ConcurrentHashMap;

/** The translator for Huggingface token classification model. */
public class TokenClassificationTranslator implements Translator<String, NamedEntity[]> {

    private HuggingFaceTokenizer tokenizer;
    private Batchifier batchifier;
    private Config config;

    TokenClassificationTranslator(HuggingFaceTokenizer tokenizer, Batchifier batchifier) {
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
    public void prepare(TranslatorContext ctx) throws IOException {
        Path path = ctx.getModel().getModelPath();
        Path file = path.resolve("config.json");
        try (Reader reader = Files.newBufferedReader(file)) {
            config = JsonUtils.GSON.fromJson(reader, Config.class);
        }
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
    public NamedEntity[] processOutput(TranslatorContext ctx, NDList list) {
        NDArray logits = list.get(0);
        Encoding encoding = (Encoding) ctx.getAttachment("encoding");
        long[] inputIds = encoding.getIds();
        CharSpan[] offsetMapping = encoding.getCharTokenSpans();
        long[] specialTokenMasks = encoding.getSpecialTokenMask();
        NDArray probabilities = logits.softmax(1);
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

    /**
     * Creates a builder to build a {@code TokenClassificationTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code TokenClassificationTranslator}.
     *
     * @param arguments the models' arguments
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = builder();
        builder.configure(arguments);

        return builder;
    }

    /** The builder for token classification translator. */
    public static final class Builder {

        private String tokenizerName;
        private Path tokenizerPath;
        private boolean addSpecialTokens = true;
        private Batchifier batchifier = Batchifier.STACK;

        /**
         * Sets the name of the tokenizer for the {@link Translator}.
         *
         * @param tokenizerName the name of the tokenizer
         * @return this builder
         */
        public Builder optTokenizerName(String tokenizerName) {
            this.tokenizerName = tokenizerName;
            return this;
        }

        /**
         * Sets the file path of the tokenizer for the {@link Translator}.
         *
         * @param tokenizerPath the path of the tokenizer
         * @return this builder
         */
        public Builder optTokenizerPath(Path tokenizerPath) {
            this.tokenizerPath = tokenizerPath;
            return this;
        }

        /**
         * Sets if add special tokens for the {@link Translator}.
         *
         * @param addSpecialTokens true to add special tokens
         * @return this builder
         */
        public Builder optAddSpecialTokens(boolean addSpecialTokens) {
            this.addSpecialTokens = addSpecialTokens;
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
            optTokenizerName(ArgumentsUtil.stringValue(arguments, "tokenizer"));
            optAddSpecialTokens(ArgumentsUtil.booleanValue(arguments, "addSpecialTokens", true));
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
            HuggingFaceTokenizer tokenizer;
            Map<String, String> options = new ConcurrentHashMap<>();
            options.put("addSpecialTokens", String.valueOf(addSpecialTokens));
            if (tokenizerName != null) {
                tokenizer = HuggingFaceTokenizer.newInstance(tokenizerName, options);
            } else {
                tokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath, options);
            }
            return new TokenClassificationTranslator(tokenizer, batchifier);
        }
    }

    private static final class Config {

        Map<String, String> id2label;
    }
}
