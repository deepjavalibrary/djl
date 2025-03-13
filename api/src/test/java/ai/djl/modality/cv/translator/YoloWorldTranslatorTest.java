/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.translator;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.VisionLanguageInput;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.YoloWorldTranslator.SimpleBpeTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.util.JsonUtils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class YoloWorldTranslatorTest {

    @Test
    public void testSimpleBpeTokenizer() throws IOException {
        Path modelPath = Paths.get("src/test/resources/yolo_world");
        SimpleBpeTokenizer tokenizer = SimpleBpeTokenizer.newInstance(modelPath);
        int[][] ids = tokenizer.batchEncode(new String[] {"cat", "remote control"});
        Assert.assertEquals(ids[0][1], 2368);
        Assert.assertEquals(ids[1][2], 3366);
        Assert.assertEquals(ids[1][3], 49407);
    }

    @Test
    public void testYoloWorldTranslator() throws ModelException, IOException, TranslateException {
        String[] candidates = {"cat", "remote control"};

        Path file = Paths.get("../examples/src/test/resources/kitten.jpg");
        Image img = ImageFactory.getInstance().fromFile(file);
        VisionLanguageInput textInput = new VisionLanguageInput(img, candidates);

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            NDArray logits = manager.ones(new Shape(1, 6, 6300));
                            return new NDList(logits);
                        },
                        "model");

        Path modelDir = Paths.get("src/test/resources/yolo_world");
        Criteria<VisionLanguageInput, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(VisionLanguageInput.class, DetectedObjects.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("threshold", "0.25")
                        .optArgument("nmsThreshold", "0.7")
                        .optArgument("clipModelPath", "identity.pt")
                        .optArgument("toTensor", true)
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new YoloWorldTranslatorFactory())
                        .build();

        try (ZooModel<VisionLanguageInput, DetectedObjects> model = criteria.loadModel();
                Predictor<VisionLanguageInput, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects.DetectedObject res = predictor.predict(textInput).best();
            Assert.assertEquals(res.getProbability(), 1.0);
            Assert.assertEquals(res.getClassName(), "cat");
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("clipModelPath", "identity.pt")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new YoloWorldTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            Map<String, Object> map = new ConcurrentHashMap<>();
            map.put("image_url", file.toUri());
            map.put("candidate_labels", textInput.getCandidates());
            input.add(JsonUtils.toJson(map));
            Output out = predictor.predict(input);
            DetectedObjects res = (DetectedObjects) out.getData().getAsObject();
            DetectedObjects.DetectedObject detection = res.best();
            Assert.assertEquals(detection.getProbability(), 1.0);
            Assert.assertEquals(detection.getClassName(), "cat");

            map.remove("candidate_labels");
            Input input1 = new Input();
            input1.add(JsonUtils.toJson(map));
            Assert.assertThrows(TranslateException.class, () -> predictor.predict(input1));
        }
    }

    @Test
    public void testYoloWorldTranslatorFactory() {
        YoloWorldTranslatorFactory factory = new YoloWorldTranslatorFactory();
        Assert.assertEquals(factory.getSupportedTypes().size(), 2);
        Map<String, String> arguments = new HashMap<>();
        try (Model model = Model.newInstance("test")) {
            Translator<VisionLanguageInput, DetectedObjects> translator1 =
                    factory.newInstance(
                            VisionLanguageInput.class, DetectedObjects.class, model, arguments);
            Assert.assertTrue(translator1 instanceof YoloWorldTranslator);

            Translator<Input, Output> translator2 =
                    factory.newInstance(Input.class, Output.class, model, arguments);
            Assert.assertTrue(translator2 instanceof ZeroShotObjectDetectionServingTranslator);

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(Image.class, Output.class, model, arguments));
        }
    }
}
