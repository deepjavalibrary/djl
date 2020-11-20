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

package ai.djl.pytorch.integration;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

/** The file is for integration test for special TorchScript input. */
public class TorchScriptTest {

    @Test
    public void testDictInput() throws ModelException, IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Criteria<NDList, NDList> criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelUrls("https://resources.djl.ai/test-models/dict_input.zip")
                            .optProgress(new ProgressBar())
                            .build();

            Path modelFile;
            try (ZooModel<NDList, NDList> model = ModelZoo.loadModel(criteria);
                    Predictor<NDList, NDList> predictor = model.newPredictor()) {
                NDArray array = manager.ones(new Shape(2, 2));
                array.setName("input1.input");
                NDList output = predictor.predict(new NDList(array));
                Assert.assertEquals(output.singletonOrThrow(), array);

                modelFile = model.getModelPath().resolve(model.getName() + ".pt");
            }

            criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelPath(modelFile)
                            .optProgress(new ProgressBar())
                            .build();

            try (ZooModel<NDList, NDList> model = ModelZoo.loadModel(criteria)) {
                Assert.assertEquals(model.getName(), "dict_input");
            }
        }
    }
}
