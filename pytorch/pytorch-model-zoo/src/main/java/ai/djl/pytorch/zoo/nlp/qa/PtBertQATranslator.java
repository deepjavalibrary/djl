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
import ai.djl.modality.nlp.bert.BertToken;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.modality.nlp.bert.BertVocabulary;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.modality.nlp.translator.QATranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The translator for {@link PtBertQATranslator}.
 *
 * @see BertQAModelLoader
 */
public class PtBertQATranslator extends QATranslator {

    private List<String> tokens;
    private BertVocabulary vocabulary;
    private BertTokenizer tokenizer;

    PtBertQATranslator() {}

    /** {@inheritDoc} */
    @Override
    public void prepare(NDManager manager, Model model) throws IOException {
        vocabulary = model.getArtifact("bert-base-uncased-vocab.txt", PtBertVocabulary::parse);
        tokenizer = new BertTokenizer(vocabulary);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input) {
        BertToken token =
                tokenizer.encode(
                        input.getQuestion().toLowerCase(), input.getParagraph().toLowerCase());

        NDManager manager = ctx.getNDManager();
        long[] indices = token.getIndices().stream().mapToLong(i -> i).toArray();
        long[] attentionMask = token.getAttentionMask().stream().mapToLong(i -> i).toArray();
        long[] tokenType = token.getTokenType().stream().mapToLong(i -> i).toArray();
        NDArray indicesArray = manager.create(indices, new Shape(1, indices.length));
        NDArray attentionMaskArray =
                manager.create(attentionMask, new Shape(1, attentionMask.length));
        NDArray tokenTypeArray = manager.create(tokenType, new Shape(1, tokenType.length));
        tokens =
                token.getIndices()
                        .stream()
                        .map(index -> vocabulary.getToken(index))
                        .collect(Collectors.toList());
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
            return new PtBertQATranslator();
        }
    }
}
