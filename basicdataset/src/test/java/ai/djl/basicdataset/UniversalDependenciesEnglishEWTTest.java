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

import ai.djl.basicdataset.nlp.UniversalDependenciesEnglishEWT;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.basicdataset.utils.TextData.Configuration;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;

public class UniversalDependenciesEnglishEWTTest {

    private static final int EMBEDDING_SIZE = 15;

    @Test
    public void testGetDataWithPreTrainedEmbedding() throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            UniversalDependenciesEnglishEWT universalDependenciesEnglishEWT =
                    UniversalDependenciesEnglishEWT.builder()
                            .optUsage(Dataset.Usage.TRAIN)
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextProcessors(
                                                    Arrays.asList(
                                                            new SimpleTokenizer(),
                                                            new LowerCaseConvertor(Locale.ENGLISH)))
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(10)
                            .build();

            universalDependenciesEnglishEWT.prepare();
            Record record = universalDependenciesEnglishEWT.get(manager, 0);
            Assert.assertEquals(record.getData().get(0).getShape().dimension(), 2);
            Assert.assertEquals(record.getLabels().get(0).getShape().dimension(), 1);
        }
    }

    @Test
    public void testGetDataWithTrainableEmbedding() throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            UniversalDependenciesEnglishEWT universalDependenciesEnglishEWT =
                    UniversalDependenciesEnglishEWT.builder()
                            .optUsage(Dataset.Usage.TEST)
                            .setSourceConfiguration(
                                    new Configuration()
                                            .setTextProcessors(
                                                    Arrays.asList(
                                                            new SimpleTokenizer(),
                                                            new LowerCaseConvertor(Locale.ENGLISH)))
                                            .setEmbeddingSize(EMBEDDING_SIZE))
                            .setSampling(32, true)
                            .optLimit(10)
                            .build();

            universalDependenciesEnglishEWT.prepare();
            Record record = universalDependenciesEnglishEWT.get(manager, 0);
            Assert.assertEquals(record.getData().size(), 1);
            Assert.assertEquals(record.getData().get(0).getShape().dimension(), 1);
            Assert.assertEquals(record.getLabels().size(), 1);
            Assert.assertEquals(record.getLabels().get(0).getShape().dimension(), 1);
        }
    }

    @Test
    public void testMisc() throws TranslateException, IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            UniversalDependenciesEnglishEWT universalDependenciesEnglishEWT =
                    UniversalDependenciesEnglishEWT.builder()
                            .optUsage(Dataset.Usage.VALIDATION)
                            .setSourceConfiguration(
                                    new TextData.Configuration()
                                            .setTextProcessors(
                                                    Arrays.asList(
                                                            new SimpleTokenizer(),
                                                            new LowerCaseConvertor(Locale.ENGLISH)))
                                            .setTextEmbedding(
                                                    TestUtils.getTextEmbedding(
                                                            manager, EMBEDDING_SIZE)))
                            .setSampling(32, true)
                            .optLimit(350)
                            .build();

            universalDependenciesEnglishEWT.prepare();
            universalDependenciesEnglishEWT.prepare();
            Assert.assertEquals(universalDependenciesEnglishEWT.size(), 350);
        }
    }
}
