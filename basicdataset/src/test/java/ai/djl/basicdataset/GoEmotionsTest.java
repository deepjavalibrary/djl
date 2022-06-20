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

import ai.djl.basicdataset.nlp.GoEmotions;
import ai.djl.basicdataset.utils.TextData;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class GoEmotionsTest {

    private static final int EMBEDDING_SIZE = 15;

    @Test
    public void testGoEmotions() throws IOException, TranslateException {
        for (Dataset.Usage usage :
                new Dataset.Usage[] {
                    Dataset.Usage.TRAIN, Dataset.Usage.VALIDATION, Dataset.Usage.TEST
                }) {
            try (NDManager manager = NDManager.newBaseManager()) {
                GoEmotions testDataSet =
                        GoEmotions.builder()
                                .setSourceConfiguration(
                                        new TextData.Configuration()
                                                .setTextEmbedding(
                                                        TestUtils.getTextEmbedding(
                                                                manager, EMBEDDING_SIZE)))
                                .optUsage(usage)
                                .setSampling(32, true)
                                .build();
                testDataSet.prepare();

                Record record = testDataSet.get(manager, 0);

                Assert.assertEquals(record.getData().size(), 1);
                Assert.assertEquals(record.getData().get(0).getShape().dimension(), 2);
                Assert.assertEquals(record.getLabels().size(), 1);
                Assert.assertEquals(record.getLabels().get(0).getShape().dimension(), 1);
            }
        }
    }
}
