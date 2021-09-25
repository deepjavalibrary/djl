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

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.onnxruntime.zoo.tabular.softmax_regression.IrisFlower;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
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
            try (ZooModel<IrisFlower, Classifications> model = criteria.loadModel();
                    Predictor<IrisFlower, Classifications> predictor = model.newPredictor()) {
                Classifications classifications = predictor.predict(virginica);
                Assert.assertEquals(classifications.best().getClassName(), "virginica");

                Model m = Model.newInstance("model", "OnnxRuntime");
                Path path = model.getModelPath();
                Assert.assertThrows(() -> m.load(path, null));
                Assert.assertThrows(() -> m.load(path, "invalid.onnx"));

                Path modelFile = path.resolve(model.getName() + ".onnx");
                m.load(modelFile);

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

    @Test
    public void testNDArray() {
        try (NDManager manager = OrtNDManager.getSystemManager().newSubManager()) {
            NDArray zeros = manager.zeros(new Shape(1, 2));
            float[] data = zeros.toFloatArray();
            Assert.assertEquals(data[0], 0);

            NDArray ones = manager.ones(new Shape(1, 2));
            data = ones.toFloatArray();
            Assert.assertEquals(data[0], 1);

            float[] buf = {0f, 1f, 2f, 3f};
            NDArray array = manager.create(buf);
            Assert.assertEquals(array.toFloatArray(), buf);

            array = manager.create("string");
            Assert.assertEquals(array.toStringArray()[0], "string");
            final NDArray a = array;
            Assert.assertThrows(IllegalArgumentException.class, a::toByteBuffer);

            array = manager.create(new String[] {"string1", "string2"});
            Assert.assertEquals(array.toStringArray()[1], "string2");
        }
    }

    @Test
    public void testStringTensor()
            throws MalformedModelException, ModelNotFoundException, IOException,
                    TranslateException {
        System.setProperty("ai.djl.onnx.disable_alternative", "true");
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optEngine("OnnxRuntime")
                        .optModelUrls(
                                "https://resources.djl.ai/test-models/onnxruntime/pipeline_tfidf.zip")
                        .build();
        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor()) {
            OrtNDManager manager = (OrtNDManager) model.getNDManager();
            NDArray stringNd =
                    manager.create(
                            new String[] {" Re: Jack can't hide from keith@cco.", " I like dogs"},
                            new Shape(1, 2));
            NDList result = predictor.predict(new NDList(stringNd));
            Assert.assertEquals(result.size(), 2);
            Assert.assertEquals(result.get(0).toLongArray(), new long[] {1});
        }
        System.clearProperty("ai.djl.onnx.disable_alternative");
    }
}
