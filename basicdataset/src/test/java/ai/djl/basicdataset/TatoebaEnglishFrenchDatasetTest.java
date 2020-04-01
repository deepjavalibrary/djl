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
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class TatoebaEnglishFrenchDatasetTest {

    @Test
    public void testGetDataWithPreTrainedEmbedding() throws IOException, EmbeddingException {
        try (NDManager manager = NDManager.newBaseManager()) {
            TatoebaEnglishFrenchDataset tatoebaEnglishFrenchDataset =
                    TatoebaEnglishFrenchDataset.builder()
                            .optSourceWordEmbedding(getWordEmbedding(manager), false)
                            .optTargetWordEmbedding(getWordEmbedding(manager), false)
                            .setTokenizer(new SimpleTokenizer())
                            .setValidLength(true)
                            .setSampling(32, true)
                            .build();
            tatoebaEnglishFrenchDataset.prepare();
            Record record = tatoebaEnglishFrenchDataset.get(manager, 0);
            Assert.assertEquals(new Shape(10, 15), record.getData().get(0).getShape());
            Assert.assertEquals(new Shape(10), record.getData().get(1).getShape());
            Assert.assertEquals(new Shape(12, 15), record.getLabels().get(0).getShape());
            Assert.assertEquals(new Shape(12), record.getLabels().get(1).getShape());
        }
    }

    @Test
    public void testGetDataWithTrainableEmbedding() throws IOException, EmbeddingException {
        try (NDManager manager = NDManager.newBaseManager()) {
            TatoebaEnglishFrenchDataset tatoebaEnglishFrenchDataset =
                    TatoebaEnglishFrenchDataset.builder()
                            .optEmbeddingSize(15)
                            .setTokenizer(new SimpleTokenizer())
                            .setValidLength(false)
                            .setSampling(32, true)
                            .build();
            tatoebaEnglishFrenchDataset.prepare();
            Record record = tatoebaEnglishFrenchDataset.get(manager, 0);
            Assert.assertEquals(new Shape(10), record.getData().get(0).getShape());
            Assert.assertEquals(record.getData().size(), 1);
            Assert.assertEquals(new Shape(12), record.getLabels().get(0).getShape());
            Assert.assertEquals(record.getLabels().size(), 1);
        }
    }

    private WordEmbedding getWordEmbedding(NDManager manager) {
        return new WordEmbedding() {
            @Override
            public boolean vocabularyContains(String word) {
                return false;
            }

            @Override
            public NDArray preprocessWordToEmbed(NDManager manager, String word) {
                return manager.zeros(new Shape());
            }

            @Override
            public NDList embedWord(ParameterStore parameterStore, NDArray word)
                    throws EmbeddingException {
                return null;
            }

            @Override
            public NDArray embedWord(NDArray word) {
                return manager.zeros(new Shape(15));
            }

            @Override
            public String unembedWord(NDArray wordEmbedding) {
                return null;
            }
        };
    }
}
