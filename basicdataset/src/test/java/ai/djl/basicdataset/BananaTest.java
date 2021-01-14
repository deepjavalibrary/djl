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

import ai.djl.basicdataset.cv.BananaDetection;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.Assertions;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.annotations.Test;

public class BananaTest {

    @Test
    public void testBananaRemote() throws IOException, TranslateException {
        BananaDetection bananaDetection =
                BananaDetection.builder()
                        .setSampling(32, false)
                        .optUsage(Dataset.Usage.TRAIN)
                        .build();

        bananaDetection.prepare();
        NDManager manager = NDManager.newBaseManager();

        for (Batch batch : bananaDetection.getData(manager)) {

            for (int i = 0; i < 1; i++) {
                NDArray imgLabel = batch.getLabels().get(0).get(i);
                Assertions.assertAlmostEquals(
                        imgLabel,
                        manager.create(
                                new float[] {0f, 0.4063f, 0.0781f, 0.5586f, 0.2266f},
                                new Shape(1, 5)));
            }
            break;
        }
    }
}
