/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class PtModelTest {

    @Test
    public void testLoadFromStream() throws IOException, TranslateException, ModelException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls("djl://ai.djl.pytorch/resnet/0.0.1/traced_resnet18")
                        .optProgress(new ProgressBar())
                        .build();
        try (ZooModel<NDList, NDList> zooModel = criteria.loadModel()) {
            Path modelFile = zooModel.getModelPath().resolve("traced_resnet18.pt");
            // This model only support CPU
            try (Model model = Model.newInstance("test model", Device.cpu())) {
                model.load(Files.newInputStream(modelFile));
                try (Predictor<NDList, NDList> predictor =
                        model.newPredictor(new NoopTranslator())) {
                    NDArray array = model.getNDManager().ones(new Shape(1, 3, 224, 224));
                    NDArray result = predictor.predict(new NDList(array)).singletonOrThrow();
                    Assert.assertEquals(result.getShape(), new Shape(1, 1000));
                }
            }

            String softMax = zooModel.getProperty("applySoftmax");
            Assert.assertEquals(softMax, "true");
            String value = zooModel.getProperty("something", "N/A");
            Assert.assertEquals(value, "N/A");
        }
    }
}
