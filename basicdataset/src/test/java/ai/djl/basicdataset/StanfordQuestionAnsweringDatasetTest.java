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
package ai.djl.basicdataset;

import ai.djl.basicdataset.nlp.StanfordQuestionAnsweringDataset;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.util.Map;

@SuppressWarnings("unchecked")
public class StanfordQuestionAnsweringDatasetTest {

    private static final int EMBEDDING_SIZE = 15;

    @Test
    public void testGetDataWithPreTrainedEmbedding() throws TranslateException, IOException {

        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(10)
                            .optUsage(Dataset.Usage.TEST)
                            .build();

            stanfordQuestionAnsweringDataset.prepare();
            Record record = stanfordQuestionAnsweringDataset.get(manager, 0);
            Assert.assertEquals(record.getData().get("title").getShape().get(0), 1);
            Assert.assertEquals(record.getData().get("question").getShape().get(0), 7);
            Assert.assertEquals(record.getLabels().size(), 4);
        }
    }

    @Test
    public void testGetDataWithTrainableEmbedding() throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration().setEmbeddingSize(EMBEDDING_SIZE))
                            .setTargetConfiguration(
                                    new TextData.Configuration().setEmbeddingSize(EMBEDDING_SIZE))
                            .setSampling(32, true)
                            .optLimit(10)
                            .build();

            stanfordQuestionAnsweringDataset.prepare();
            Record record = stanfordQuestionAnsweringDataset.get(manager, 0);
            Assert.assertEquals(record.getData().get("title").getShape().dimension(), 1);
            Assert.assertEquals(record.getData().get("context").getShape().get(0), 156);
            Assert.assertEquals(record.getLabels().size(), 1);
        }
    }

    @Test
    public void testInvalidUsage() throws TranslateException, IOException {

        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(10)
                            .optUsage(Dataset.Usage.VALIDATION)
                            .build();

            stanfordQuestionAnsweringDataset.prepare();
        } catch (UnsupportedOperationException uoe) {
            Assert.assertEquals(uoe.getMessage(), "Validation data not available.");
        }
    }

    @Test
    public void testMisc() throws TranslateException, IOException {

        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optUsage(Dataset.Usage.TEST)
                            .build();

            stanfordQuestionAnsweringDataset.prepare();
            stanfordQuestionAnsweringDataset.prepare();
            Assert.assertEquals(stanfordQuestionAnsweringDataset.size(), 11873);

            Record record0 = stanfordQuestionAnsweringDataset.get(manager, 0);
            Record record6 = stanfordQuestionAnsweringDataset.get(manager, 6);
            Assert.assertEquals(record6.getData().get("title").getShape().dimension(), 2);
            Assert.assertEquals(
                    record0.getData().get("context").getShape().get(0),
                    record6.getData().get("context").getShape().get(0));
            Assert.assertEquals(record6.getLabels().size(), 0);
        }
    }

    @Test
    public void testLimitBoundary() throws TranslateException, IOException {

        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(3)
                            .optUsage(Dataset.Usage.TEST)
                            .build();

            stanfordQuestionAnsweringDataset.prepare();
            Assert.assertEquals(stanfordQuestionAnsweringDataset.size(), 3);
            Record record = stanfordQuestionAnsweringDataset.get(manager, 2);
            Assert.assertEquals(record.getData().get("title").getShape().dimension(), 2);
            Assert.assertEquals(record.getData().get("context").getShape().get(0), 140);
            Assert.assertEquals(record.getLabels().size(), 4);
        }
    }

    @Test
    public void testRawData() throws IOException {

        try (NDManager manager = NDManager.newBaseManager()) {
            StanfordQuestionAnsweringDataset stanfordQuestionAnsweringDataset =
                    StanfordQuestionAnsweringDataset.builder()
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setTargetConfiguration(
                                    new TextData.Configuration()
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(350)
                            .optUsage(Dataset.Usage.TEST)
                            .build();

            Map<String, Object> data =
                    (Map<String, Object>) stanfordQuestionAnsweringDataset.getData();
            Assert.assertEquals(data.get("version").toString(), "v2.0");
        }
    }
}
