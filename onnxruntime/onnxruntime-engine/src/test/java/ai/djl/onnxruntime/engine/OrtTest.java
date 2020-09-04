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

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class OrtTest {

    @Test
    public void testOrt() throws TranslateException, ModelException, IOException {
        try {
            String modelUrl = "https://alpha-djl-demos.s3.amazonaws.com/model/sklearn/rf_iris.zip";
            Criteria<FlowerInfo, Classifications> criteria =
                    Criteria.builder()
                            .setTypes(FlowerInfo.class, Classifications.class)
                            .optModelUrls(modelUrl)
                            .optTranslator(new MyTranslator())
                            .optEngine("OnnxRuntime") // use OnnxRuntime engine
                            .build();

            FlowerInfo virginica = new FlowerInfo(1.0f, 2.0f, 3.0f, 4.0f);
            try (ZooModel<FlowerInfo, Classifications> model = ModelZoo.loadModel(criteria);
                    Predictor<FlowerInfo, Classifications> predictor = model.newPredictor()) {
                Classifications classifications = predictor.predict(virginica);
                Assert.assertEquals(classifications.best().getClassName(), "virginica");
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

    private static final class FlowerInfo {

        float sepalLength;
        float sepalWidth;
        float petalLength;
        float petalWidth;

        public FlowerInfo(
                float sepalLength, float sepalWidth, float petalLength, float petalWidth) {
            this.sepalLength = sepalLength;
            this.sepalWidth = sepalWidth;
            this.petalLength = petalLength;
            this.petalWidth = petalWidth;
        }
    }

    private static class MyTranslator implements Translator<FlowerInfo, Classifications> {

        private final List<String> synset;

        public MyTranslator() {
            // species name
            synset = Arrays.asList("setosa", "versicolor", "virginica");
        }

        @Override
        public NDList processInput(TranslatorContext ctx, FlowerInfo input) {
            float[] data = {
                input.sepalLength, input.sepalWidth, input.petalLength, input.petalWidth
            };
            NDArray array = ctx.getNDManager().create(data, new Shape(1, 4));
            return new NDList(array);
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            return new Classifications(synset, list.get(1));
        }

        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }
}
