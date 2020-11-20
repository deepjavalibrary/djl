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
package ai.djl.onnxruntime.engine;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.onnxruntime.zoo.tabular.randomforest.IrisFlower;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class OrtTest {

    @Test
    public void testOrt() throws TranslateException, ModelException, IOException {
        try {
            Criteria<IrisFlower, Classifications> criteria =
                    Criteria.builder()
                            .setTypes(IrisFlower.class, Classifications.class)
                            .optEngine("OnnxRuntime") // use OnnxRuntime engine
                            .build();

            IrisFlower virginica = new IrisFlower(1.0f, 2.0f, 3.0f, 4.0f);
            try (ZooModel<IrisFlower, Classifications> model = ModelZoo.loadModel(criteria);
                    Predictor<IrisFlower, Classifications> predictor = model.newPredictor()) {
                Classifications classifications = predictor.predict(virginica);
                Assert.assertEquals(classifications.best().getClassName(), "virginica");

                Model m = Model.newInstance("model", "OnnxRuntime");
                Path path = model.getModelPath();
                Assert.assertThrows(() -> m.load(path, null));
                Assert.assertThrows(() -> m.load(path, "invalid.onnx"));

                Path modleFile = path.resolve(model.getName() + ".onnx");
                m.load(modleFile);

                m.close();
            }
        } catch (UnsatisfiedLinkError e) {
            /*
             * FIXME: Ort requires libgomp.so.1 pre-installed, we should manually copy
             * libgomp.so.1 to our cache folder and set "onnxruntime.native." + library + ".path"
             * to djl cache directory.
             */
            throw new SkipException("Ignore missing libgomp.so.1 error.");
        }
    }
}
