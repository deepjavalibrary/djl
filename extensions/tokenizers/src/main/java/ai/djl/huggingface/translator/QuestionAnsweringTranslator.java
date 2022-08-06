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
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** The translator for Huggingface question answering model. */
public class QuestionAnsweringTranslator implements Translator<QAInput, String> {

    private HuggingFaceTokenizer tokenizer;
    private boolean includeTokenTypes;
    private Batchifier batchifier;

    QuestionAnsweringTranslator(
            HuggingFaceTokenizer tokenizer, boolean includeTokenTypes, Batchifier batchifier) {
        this.tokenizer = tokenizer;
        this.includeTokenTypes = includeTokenTypes;
        this.batchifier = batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input) {
        NDManager manager = ctx.getNDManager();
        Encoding encoding = tokenizer.encode(input.getQuestion(), input.getParagraph());
        ctx.setAttachment("encoding", encoding);
        long[] indices = encoding.getIds();
        long[] attentionMask = encoding.getAttentionMask();
        NDList ndList = new NDList(3);
        ndList.add(manager.create(indices));
        ndList.add(manager.create(attentionMask));
        if (includeTokenTypes) {
            long[] typeIds = encoding.getTypeIds();
            ndList.add(manager.create(typeIds));
        }
        return ndList;
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        NDArray startLogits = list.get(0).duplicate();
        NDArray endLogits = list.get(1).duplicate();
        startLogits.set(new NDIndex(0), 0);
        endLogits.set(new NDIndex(0), 0);
        int startIdx = (int) startLogits.argMax().getLong();
        int endIdx = (int) endLogits.argMax().getLong();
        if (startIdx > endIdx) {
            int tmp = startIdx;
            startIdx = endIdx;
            endIdx = tmp;
        }
        Encoding encoding = (Encoding) ctx.getAttachment("encoding");
        long[] indices = encoding.getIds();
        int len = endIdx - startIdx + 1;
        long[] ids = new long[len];
        System.arraycopy(indices, startIdx, ids, 0, len);
        return tokenizer.decode(ids).trim();
    }

    /**
     * Creates a builder to build a {@code QuestionAnsweringTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code QuestionAnsweringTranslator}.
     *
     * @param arguments the models' arguments
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = builder();
        builder.configure(arguments);

        return builder;
    }

    /** The builder for question answering translator. */
    public static final class Builder {

        private String tokenizerName;
        private Path tokenizerPath;
        private boolean includeTokenTypes;
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
         * @param tokenizerPath the name of the tokenizer
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
         * Sets the {@link Batchifier} for the {@link Translator}.
         *
         * @param batchifier true to include token types
         * @return this builder
         */
        public Builder optBatchifer(Batchifier batchifier) {
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
            optIncludeTokenTypes(ArgumentsUtil.booleanValue(arguments, "includeTokenTypes"));
            optAddSpecialTokens(ArgumentsUtil.booleanValue(arguments, "addSpecialTokens", true));
            optBatchifer(
                    Batchifier.fromString(
                            ArgumentsUtil.stringValue(arguments, "batchifier", "stack")));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException if I/O error occurs
         */
        public QuestionAnsweringTranslator build() throws IOException {
            HuggingFaceTokenizer tokenizer;
            Map<String, String> options = new ConcurrentHashMap<>();
            options.put("addSpecialTokens", String.valueOf(addSpecialTokens));
            if (tokenizerName != null) {
                tokenizer = HuggingFaceTokenizer.newInstance(tokenizerName, options);
            } else {
                tokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath, options);
            }
            return new QuestionAnsweringTranslator(tokenizer, includeTokenTypes, batchifier);
        }
    }
}
