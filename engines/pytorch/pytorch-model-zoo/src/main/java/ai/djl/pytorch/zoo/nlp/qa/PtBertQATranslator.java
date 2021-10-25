/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.zoo.nlp.qa;

import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertFullTokenizer;
import ai.djl.modality.nlp.bert.BertToken;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.modality.nlp.translator.QATranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/** The {@link ai.djl.translate.Translator} for PyTorch Question Answering model. */
public class PtBertQATranslator extends QATranslator {

    private List<String> tokens;
    private Vocabulary vocabulary;
    private BertTokenizer tokenizer;

    PtBertQATranslator(Builder builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        vocabulary =
                DefaultVocabulary.builder()
                        .addFromTextFile(ctx.getModel().getArtifact(vocab))
                        .optUnknownToken("[UNK]")
                        .build();
        if (tokenizerName == null) {
            tokenizer = new BertTokenizer();
        } else {
            tokenizer = new BertFullTokenizer(vocabulary, true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input) {
        String question = input.getQuestion();
        String paragraph = input.getParagraph();
        if (toLowerCase) {
            question = question.toLowerCase(locale);
            paragraph = paragraph.toLowerCase(locale);
        }
        BertToken token = tokenizer.encode(question, paragraph);
        tokens = token.getTokens();
        NDManager manager = ctx.getNDManager();
        long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();
        long[] attentionMask = token.getAttentionMask().stream().mapToLong(i -> i).toArray();
        NDList ndList = new NDList(3);
        ndList.add(manager.create(indices));
        ndList.add(manager.create(attentionMask));
        if (includeTokenTypes) {
            long[] tokenTypes = token.getTokenTypes().stream().mapToLong(i -> i).toArray();
            ndList.add(manager.create(tokenTypes));
        }
        return ndList;
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        NDArray startLogits = list.get(0);
        NDArray endLogits = list.get(1);
        int startIdx = (int) startLogits.argMax().getLong();
        int endIdx = (int) endLogits.argMax().getLong();
        if (startIdx >= endIdx) {
            return "";
        }
        return tokenizer.tokenToString(tokens.subList(startIdx, endIdx + 1));
    }

    /**
     * Creates a builder to build a {@code PtBertQATranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code PtSSDTranslatorBuilder} with specified arguments.
     *
     * @param arguments arguments to specify builder options
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = new Builder();
        builder.configure(arguments);
        return builder;
    }

    /** The builder for Bert QA translator. */
    public static class Builder extends BaseBuilder<Builder> {

        /**
         * Returns the builder.
         *
         * @return the builder
         */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        protected PtBertQATranslator build() {
            return new PtBertQATranslator(this);
        }
    }
}
