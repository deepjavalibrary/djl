/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.mxnet.zoo.nlp.qa;

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
import ai.djl.util.Utils;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The translator for {@link BertQAModelLoader}.
 *
 * @see BertQAModelLoader
 */
public class MxBertQATranslator extends QATranslator {
    private List<String> tokens;
    private BertVocabulary vocabulary;
    private BertTokenizer tokenizer;
    private int seqLength;

    MxBertQATranslator(Builder builder) {
        seqLength = builder.seqLength;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(NDManager manager, Model model) throws IOException {
        vocabulary = model.getArtifact("vocab.json", MxBertVocabulary::parse);
        tokenizer = new BertTokenizer(vocabulary);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input) {
        BertToken token =
                tokenizer.encode(
                        input.getQuestion().toLowerCase(),
                        input.getParagraph().toLowerCase(),
                        seqLength);
        float[] indexesFloat = Utils.toFloatArray(token.getIndices());
        float[] types = Utils.toFloatArray(token.getTokenType());
        int validLength = token.getValidLength();
        tokens =
                token.getIndices()
                        .stream()
                        .map(index -> vocabulary.getToken(index))
                        .collect(Collectors.toList());

        NDManager manager = ctx.getNDManager();
        NDArray data0 = manager.create(indexesFloat, new Shape(1, seqLength));
        data0.setName("data0");
        NDArray data1 = manager.create(types, new Shape(1, seqLength));
        data1.setName("data1");
        NDArray data2 = manager.create(new float[] {validLength});
        data2.setName("data2");

        return new NDList(data0, data1, data2);
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        NDArray array = list.singletonOrThrow();
        NDList output = array.split(2, 2);
        // Get the formatted logits result
        NDArray startLogits = output.get(0).reshape(new Shape(1, -1));
        NDArray endLogits = output.get(1).reshape(new Shape(1, -1));
        int startIdx = (int) startLogits.argMax(1).getLong();
        int endIdx = (int) endLogits.argMax(1).getLong();
        return tokens.subList(startIdx, endIdx + 1).toString();
    }

    /**
     * Creates a builder to build a {@code MxBertQATranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for Bert QA translator. */
    public static class Builder extends BaseBuilder<Builder> {
        private int seqLength;

        /**
         * Set the max length of the sequence to do the padding.
         *
         * @param seqLength the length of the sequence
         * @return builder
         */
        public Builder setSeqLength(int seqLength) {
            this.seqLength = seqLength;
            return self();
        }

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
        protected MxBertQATranslator build() {
            if (seqLength == 0) {
                throw new IllegalArgumentException("You must specify a seqLength with value > 0");
            }
            return new MxBertQATranslator(this);
        }
    }
}
