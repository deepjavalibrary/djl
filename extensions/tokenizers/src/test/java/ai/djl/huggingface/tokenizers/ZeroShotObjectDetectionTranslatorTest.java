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
package ai.djl.huggingface.tokenizers;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.huggingface.translator.ZeroShotObjectDetectionTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.VisionLanguageInput;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.Assertions;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ZeroShotObjectDetectionTranslatorTest {

    @Test
    public void testZeroShotObjectDetectionTranslator()
            throws ModelException, IOException, TranslateException {
        Path file = Paths.get("../../examples/src/test/resources/kitten.jpg");
        Image img = ImageFactory.getInstance().fromFile(file);
        VisionLanguageInput textInput = new VisionLanguageInput(img, new String[] {"a cat"});

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            NDArray logits = manager.zeros(new Shape(1, 3600, 1));
                            logits.setName("logits");
                            logits.set(new NDIndex(0), -4);
                            logits.set(new NDIndex(0, 0, 0), 2);
                            NDArray boxes = manager.zeros(new Shape(1, 3600, 4));
                            boxes.setName("pred_boxes");
                            boxes.set(new NDIndex(0, 0, 0), 0.76164067);
                            boxes.set(new NDIndex(0, 0, 1), 0.28735778);
                            boxes.set(new NDIndex(0, 0, 2), 0.48600525);
                            boxes.set(new NDIndex(0, 0, 3), 0.50610954);
                            return new NDList(logits, boxes);
                        },
                        "model");

        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);

        Criteria<VisionLanguageInput, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(VisionLanguageInput.class, DetectedObjects.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("width", 960)
                        .optArgument("height", 960)
                        .optArgument("pad", 128)
                        .optArgument("resize", "true")
                        .optArgument("threshold", "0.2")
                        .optArgument("tokenizer", "google-bert/bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optArgument("toTensor", true)
                        .optTranslatorFactory(new ZeroShotObjectDetectionTranslatorFactory())
                        .build();

        try (ZooModel<VisionLanguageInput, DetectedObjects> model = criteria.loadModel();
                Predictor<VisionLanguageInput, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects.DetectedObject res = predictor.predict(textInput).best();
            Assertions.assertAlmostEquals(res.getProbability(), 0.88079);
            Assert.assertEquals(res.getClassName(), "a cat");
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "google-bert/bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new ZeroShotObjectDetectionTranslatorFactory())
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
            Assertions.assertAlmostEquals(detection.getProbability(), 0.88079);
            Assert.assertEquals(detection.getClassName(), "a cat");

            map.remove("candidate_labels");
            Input input1 = new Input();
            input1.add(JsonUtils.toJson(map));
            Assert.assertThrows(TranslateException.class, () -> predictor.predict(input1));
        }

        try (Model model = Model.newInstance("test")) {
            model.setBlock(block);
            Map<String, String> options = new HashMap<>();
            options.put("hasParameter", "false");
            model.load(modelDir, "test", options);

            ZeroShotObjectDetectionTranslatorFactory factory =
                    new ZeroShotObjectDetectionTranslatorFactory();
            Map<String, String> arguments = new HashMap<>();

            Assert.assertThrows(
                    TranslateException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));

            arguments.put("tokenizer", "google-bert/bert-base-uncased");

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));
        }
    }
}
