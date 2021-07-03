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
package ai.djl.pytorch.zoo.nlp.sentimentanalysis;

import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.modality.nlp.SimpleVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.StackBatchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;

/** The {@link ai.djl.translate.Translator} for PyTorch Sentiment Analysis model. */
public class PtDistilBertTranslator implements Translator<String, Classifications> {

    private Vocabulary vocabulary;
    private BertTokenizer tokenizer;

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return new StackBatchifier();
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(NDManager manager, Model model) throws IOException {
        URL url = model.getArtifact("distilbert-base-uncased-finetuned-sst-2-english-vocab.txt");
        vocabulary =
                SimpleVocabulary.builder()
                        .optMinFrequency(1)
                        .addFromTextFile(url)
                        .optUnknownToken("[UNK]")
                        .build();
        tokenizer = new BertTokenizer();
    }

    /** {@inheritDoc} */
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray raw = list.singletonOrThrow();
        NDArray computed = raw.exp().div(raw.exp().sum(new int[] {0}, true));
        return new Classifications(Arrays.asList("Negative", "Positive"), computed);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        List<String> tokens = tokenizer.tokenize(input);
        long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();
        long[] attentionMask = new long[tokens.size()];
        Arrays.fill(attentionMask, 1);
        NDManager manager = ctx.getNDManager();
        NDArray indicesArray = manager.create(indices);
        NDArray attentionMaskArray = manager.create(attentionMask);
        return new NDList(indicesArray, attentionMaskArray);
    }
}
