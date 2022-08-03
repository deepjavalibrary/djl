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

package ai.djl.ml.lightgbm;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.TestRequirements;
import ai.djl.training.util.DownloadUtils;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class LgbmModelTest {

    @Test
    public void testLoad() throws MalformedModelException, IOException, TranslateException {
        TestRequirements.notArm();
        Path modelDir = Paths.get("build/model");
        DownloadUtils.download(
                "https://resources.djl.ai/test-models/lightgbm/quadratic.txt",
                modelDir.resolve("quadratic.txt").toString());

        try (Model model = Model.newInstance("model")) {
            model.load(modelDir, "quadratic");
            Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator());
            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray array = manager.ones(new Shape(10, 4));
                NDList output = predictor.predict(new NDList(array));
                Assert.assertEquals(output.singletonOrThrow().getDataType(), DataType.FLOAT64);
                Assert.assertEquals(output.singletonOrThrow().getShape().size(), 10);
            }
        }
    }
}
