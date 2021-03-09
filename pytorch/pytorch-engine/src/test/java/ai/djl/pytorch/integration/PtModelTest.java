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

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtModel;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.net.URL;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PtModelTest {

    @Test
    public void testLoadFromStream() throws IOException, TranslateException {
        URL url =
                new URL("https://djl-ai.s3.amazonaws.com/resources/test-models/traced_resnet18.pt");
        try (PtModel model = (PtModel) Model.newInstance("test model")) {
            model.load(url.openStream());
            try (Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator())) {
                NDArray array = model.getNDManager().ones(new Shape(1, 3, 224, 224));
                NDArray result = predictor.predict(new NDList(array)).singletonOrThrow();
                Assert.assertEquals(result.getShape(), new Shape(1, 1000));
            }
        }
    }
}
