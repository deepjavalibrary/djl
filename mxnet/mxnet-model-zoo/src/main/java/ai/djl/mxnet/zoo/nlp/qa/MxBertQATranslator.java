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
import ai.djl.modality.nlp.SimpleVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertToken;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.modality.nlp.translator.QATranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;
import com.google.gson.annotations.SerializedName;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Collectors;

/** The translator for MXNet BERT QA model. */
public class MxBertQATranslator extends QATranslator {

    private List<String> tokens;
    private Vocabulary vocabulary;
    private BertTokenizer tokenizer;
    private int seqLength;

    MxBertQATranslator(Builder builder) {
        super(builder);
        seqLength = builder.seqLength;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(NDManager manager, Model model) throws IOException {
        vocabulary =
                SimpleVocabulary.builder()
                        .optMinFrequency(1)
                        .addFromCustomizedFile(
                                model.getArtifact("vocab.json"), VocabParser::parseToken)
                        .optUnknownToken("[UNK]")
                        .build();
        tokenizer = new BertTokenizer();
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        // MXNet BertQA model doesn't support batch
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input) {
        BertToken token =
                tokenizer.encode(
                        input.getQuestion().toLowerCase(),
                        input.getParagraph().toLowerCase(),
                        seqLength);
        tokens = token.getTokens();
        List<Long> indices =
                token.getTokens().stream().map(vocabulary::getIndex).collect(Collectors.toList());
        float[] indexesFloat = Utils.toFloatArray(indices);
        float[] types = Utils.toFloatArray(token.getTokenTypes());
        int validLength = token.getValidLength();

        NDManager manager = ctx.getNDManager();
        NDArray data0 = manager.create(indexesFloat);
        data0.setName("data0");
        NDArray data1 = manager.create(types);
        data1.setName("data1");
        // avoid to use scalar as MXNet Bert model was trained with 1.5.0
        // which is not compatible with MXNet NumPy
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

    private static final class VocabParser {

        @SerializedName("idx_to_token")
        List<String> idx2token;

        public static List<String> parseToken(URL url) {
            try (InputStream is = url.openStream();
                    Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
                return JsonUtils.GSON.fromJson(reader, VocabParser.class).idx2token;
            } catch (IOException e) {
                throw new IllegalArgumentException("Invalid url: " + url, e);
            }
        }
    }
}
