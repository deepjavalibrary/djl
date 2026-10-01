/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Point;
import ai.djl.modality.cv.translator.Sam2Translator.Sam2Input;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class Sam2TranslatorFactoryTest {

    @Test
    public void testNewInstance() {
        Sam2TranslatorFactory factory = new Sam2TranslatorFactory();
        Assert.assertEquals(factory.getSupportedTypes().size(), 2);
        Map<String, String> arguments = new HashMap<>();
        try (Model model = Model.newInstance("test")) {
            Translator<Sam2Input, DetectedObjects> translator1 =
                    factory.newInstance(Sam2Input.class, DetectedObjects.class, model, arguments);
            Assert.assertTrue(translator1 instanceof Sam2Translator);

            Translator<Input, Output> translator5 =
                    factory.newInstance(Input.class, Output.class, model, arguments);
            Assert.assertTrue(translator5 instanceof Sam2ServingTranslator);

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(Image.class, Output.class, model, arguments));
        }
    }

    @Test
    public void testEncoderOutsideModelDir()
            throws ModelException, IOException, TranslateException {
        if (Utils.isFileOutsideModelDirAllowed()) {
            throw new SkipException("Files outside the model directory are allowed");
        }
        Path file = Paths.get("../examples/src/test/resources/kitten.jpg");
        Image img = ImageFactory.getInstance().fromFile(file);
        Sam2Input input = new Sam2Input(img, new Point[] {new Point(10, 10)}, new int[] {1});
        Block block = new LambdaBlock(a -> a, "model");

        // An encoder next to the model directory, rather than in it.
        Path root = Files.createTempDirectory("djl-sam2");
        try {
            Path modelDir = Files.createDirectories(root.resolve("model"));
            Path encoder =
                    Files.copy(
                            Paths.get("src/test/resources/yolo_world/identity.pt"),
                            root.resolve("encoder.pt"));
            for (String name :
                    new String[] {"../encoder.pt", encoder.toAbsolutePath().toString()}) {
                Criteria<Sam2Input, DetectedObjects> criteria =
                        Criteria.builder()
                                .setTypes(Sam2Input.class, DetectedObjects.class)
                                .optModelPath(modelDir)
                                .optBlock(block)
                                .optEngine("PyTorch")
                                .optArgument("encoder", name)
                                .optOption("hasParameter", "false")
                                .optTranslatorFactory(new Sam2TranslatorFactory())
                                .build();
                try (ZooModel<Sam2Input, DetectedObjects> model = criteria.loadModel();
                        Predictor<Sam2Input, DetectedObjects> predictor = model.newPredictor()) {
                    TranslateException e =
                            Assert.expectThrows(
                                    TranslateException.class, () -> predictor.predict(input));
                    Assert.assertTrue(
                            e.getCause() instanceof IllegalArgumentException,
                            name + " -> " + e.getCause());
                }
            }
        } finally {
            Utils.deleteQuietly(root);
        }
    }
}
