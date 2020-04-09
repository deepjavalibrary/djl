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
package ai.djl.basicdataset;

import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.modality.nlp.embedding.WordEmbedding;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class StanfordMovieReviewTest {

    private static final int EMBEDDING_SIZE = 15;

    @Test
    public void testGetDataWithPreTrainedEmbedding() throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordMovieReview dataset =
                    StanfordMovieReview.builder()
                            .optSourceWordEmbedding(getWordEmbedding(manager), false)
                            .optTargetWordEmbedding(getWordEmbedding(manager), false)
                            .setTokenizer(new SimpleTokenizer())
                            .setValidLength(true)
                            .setSampling(32, true)
                            .build();
            dataset.prepare();

            Record record = dataset.get(manager, 0);
            Assert.assertEquals(record.getData().get(0).getShape().dimension(), 2);
            Assert.assertEquals(record.getData().get(0).getShape().get(1), EMBEDDING_SIZE);
            Assert.assertEquals(record.getLabels().get(0).getShape().dimension(), 0);
        }
    }

    @Test
    public void testGetDataWithTrainableEmbedding() throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordMovieReview dataset =
                    StanfordMovieReview.builder()
                            .optEmbeddingSize(EMBEDDING_SIZE)
                            .setTokenizer(new SimpleTokenizer())
                            .setValidLength(false)
                            .setSampling(32, true)
                            .build();
            dataset.prepare();

            Record record = dataset.get(manager, 0);
            Assert.assertEquals(record.getData().get(0).getShape().dimension(), 1);
            Assert.assertEquals(record.getLabels().get(0).getShape().dimension(), 0);
        }
    }

    private WordEmbedding getWordEmbedding(NDManager manager) {
        return new WordEmbedding() {

            /** {@inheritDoc} */
            @Override
            public boolean vocabularyContains(String word) {
                return false;
            }

            /** {@inheritDoc} */
            @Override
            public NDArray preprocessWordToEmbed(NDManager manager, String word) {
                return manager.zeros(new Shape());
            }

            /** {@inheritDoc} */
            @Override
            public NDList embedWord(ParameterStore parameterStore, NDArray word)
                    throws EmbeddingException {
                return null;
            }

            /** {@inheritDoc} */
            @Override
            public NDArray embedWord(NDArray word) {
                return manager.zeros(new Shape(EMBEDDING_SIZE));
            }

            /** {@inheritDoc} */
            @Override
            public String unembedWord(NDArray wordEmbedding) {
                return null;
            }
        };
    }
}
