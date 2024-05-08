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
package ai.djl.audio.dataset;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class LibrispeechTest {

    @Test
    public void testLibrispeech() throws IOException, TranslateException {
        // The dataset is very large, only run the test when -Dweekly=true is set
        if (Boolean.getBoolean("weekly")) {
            NDManager manager = NDManager.newBaseManager();
            Librispeech dataset =
                    Librispeech.builder()
                            .optUsage(Dataset.Usage.TEST)
                            .setSampling(32, false)
                            .build();
            dataset.prepare();
            NDList data = dataset.get(manager, 0).getData();
            Assert.assertEquals(data.getShapes()[0].getShape(), new long[] {161, 306});

            long[] expected = {
                0, 1, 2, 3, 2344, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2, 14, 15, 16, 5, 2344, 17, 18,
                19
            };
            NDList labels = dataset.get(manager, 0).getLabels();
            Assert.assertEquals(labels.get(0).toLongArray(), expected);
        }
    }
}
