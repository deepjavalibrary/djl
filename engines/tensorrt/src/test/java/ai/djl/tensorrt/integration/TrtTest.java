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

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.tensorrt.engine.TrtSession;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.cuda.CudaUtils;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class TrtTest {

    @Test
    public void testTrtOnnx() throws ModelException, IOException, TranslateException {
        Engine engine;
        try {
            engine = Engine.getEngine("TensorRT");
        } catch (Exception ignore) {
            throw new SkipException("Your os configuration doesn't support TensorRT.");
        }
        if (!engine.defaultDevice().isGpu()) {
            throw new SkipException("TensorRT only support GPU.");
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

    @Test
    public void testTrtUff() throws ModelException, IOException, TranslateException {
        Engine engine;
        try {
            engine = Engine.getEngine("TensorRT");
        } catch (Exception ignore) {
            throw new SkipException("Your os configuration doesn't support TensorRT.");
        }
        if (!engine.defaultDevice().isGpu()) {
            throw new SkipException("TensorRT only support GPU.");
        }
        List<String> synset =
                IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
        ImageClassificationTranslator translator =
                ImageClassificationTranslator.builder()
                        .optFlag(Image.Flag.GRAYSCALE)
                        .optSynset(synset)
                        .optApplySoftmax(true)
                        .addTransform(new ToTensor())
                        .optBatchifier(null)
                        .build();
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelUrls("https://resources.djl.ai/test-models/tensorrt/lenet5.zip")
                        .optTranslator(translator)
                        .optEngine("TensorRT")
                        .build();
        try (ZooModel<Image, Classifications> model = criteria.loadModel();
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            Path path = Paths.get("../../examples/src/test/resources/0.png");
            Image image = ImageFactory.getInstance().fromFile(path);
            Classifications ret = predictor.predict(image);
            Assert.assertEquals(ret.best().getClassName(), "0");
        }
    }

    @Test
    public void testSerializedEngine() throws ModelException, IOException, TranslateException {
        Engine engine;
        try {
            engine = Engine.getEngine("TensorRT");
        } catch (Exception ignore) {
            throw new SkipException("Your os configuration doesn't support TensorRT.");
        }
        Device device = engine.defaultDevice();
        if (!device.isGpu()) {
            throw new SkipException("TensorRT only support GPU.");
        }
        String sm = CudaUtils.getComputeCapability(device.getDeviceId());
        Criteria<float[], float[]> criteria =
                Criteria.builder()
                        .setTypes(float[].class, float[].class)
                        .optModelPath(Paths.get("src/test/resources/identity_" + sm + ".trt"))
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

    private static final class MyTranslator implements NoBatchifyTranslator<float[], float[]> {

        private TrtSession session;

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
