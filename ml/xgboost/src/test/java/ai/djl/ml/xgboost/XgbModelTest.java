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

package ai.djl.ml.xgboost;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.util.DownloadUtils;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class XgbModelTest {

    @BeforeClass
    public void downloadXGBoostModel() throws IOException {
        Path modelDir = Paths.get("build/model");
        DownloadUtils.download(
                "https://resources.djl.ai/test-models/xgboost/regression.json",
                modelDir.resolve("regression.json").toString());
    }

    @Test
    public void testLoad() throws MalformedModelException, IOException, TranslateException {
        // TODO: skip for Windows since it is not supported by XGBoost
        if (!System.getProperty("os.name").startsWith("Win")) {
            try (Model model = Model.newInstance("XGBoost")) {
                model.load(Paths.get("build/model"), "regression");
                Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator());
                try (NDManager manager = NDManager.newBaseManager()) {
                    NDArray array = manager.ones(new Shape(10, 13));
                    NDList output = predictor.predict(new NDList(array));
                    Assert.assertEquals(output.singletonOrThrow().toFloatArray().length, 10);
                }
            }
        }
    }
}
