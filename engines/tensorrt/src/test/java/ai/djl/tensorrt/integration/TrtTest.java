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
package ai.djl.tensorrt.integration;

import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.tensorrt.engine.TrtSession;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class TrtTest {

    @Test
    public void testTrtOnnx() throws ModelException, IOException, TranslateException {
        try {
            Engine.getEngine("TensorRT");
        } catch (Exception ignore) {
            throw new SkipException("Your os configuration doesn't support TensorRT.");
        }
        Criteria<float[], float[]> criteria =
                Criteria.builder()
                        .setTypes(float[].class, float[].class)
                        .optModelPath(Paths.get("src/test/resources/identity.onnx"))
                        .optTranslator(new MyTranslator())
                        .optEngine("TensorRT")
                        .build();
        try (ZooModel<float[], float[]> model = criteria.loadModel();
                Predictor<float[], float[]> predictor = model.newPredictor()) {
            float[] data = new float[] {1, 2, 3, 4};
            float[] ret = predictor.predict(data);
            Assert.assertEquals(ret, data);
        }
    }

    private static final class MyTranslator implements Translator<float[], float[]> {

        private TrtSession session;

        /** {@inheritDoc} */
        @Override
        public Batchifier getBatchifier() {
            return null;
        }

        /** {@inheritDoc} */
        @Override
        public void prepare(TranslatorContext ctx) {
            session = (TrtSession) ctx.getBlock();
        }

        /** {@inheritDoc} */
        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            return list.head().toFloatArray();
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, float[] input) {
            // Reuse NDArrays that bound to TensorRT engine
            NDList inputs = session.getInputBindings();
            inputs.head().set(FloatBuffer.wrap(input));
            return inputs;
        }
    }
}
