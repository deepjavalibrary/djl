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

import ai.djl.Model;
import ai.djl.modality.nlp.SimpleVocabulary;
import ai.djl.modality.nlp.Vocabulary;
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
    public void prepare(NDManager manager, Model model) throws IOException {
        vocabulary =
                SimpleVocabulary.builder()
                        .optMinFrequency(1)
                        .addFromTextFile(model.getArtifact("bert-base-uncased-vocab.txt"))
                        .optUnknownToken("[UNK]")
                        .build();
        tokenizer = new BertTokenizer();
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input) {
        BertToken token =
                tokenizer.encode(
                        input.getQuestion().toLowerCase(), input.getParagraph().toLowerCase());
        tokens = token.getTokens();
        NDManager manager = ctx.getNDManager();
        long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();
        long[] attentionMask = token.getAttentionMask().stream().mapToLong(i -> i).toArray();
        long[] tokenType = token.getTokenTypes().stream().mapToLong(i -> i).toArray();
        NDArray indicesArray = manager.create(indices);
        NDArray attentionMaskArray = manager.create(attentionMask);
        NDArray tokenTypeArray = manager.create(tokenType);
        return new NDList(indicesArray, attentionMaskArray, tokenTypeArray);
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        NDArray startLogits = list.get(0);
        NDArray endLogits = list.get(1);
        int startIdx = (int) startLogits.argMax().getLong();
        int endIdx = (int) endLogits.argMax().getLong();
        return tokens.subList(startIdx, endIdx + 1).toString();
    }

    /**
     * Creates a builder to build a {@code PtBertQATranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
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
