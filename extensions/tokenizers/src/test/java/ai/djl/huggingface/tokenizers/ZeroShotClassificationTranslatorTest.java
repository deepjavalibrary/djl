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
import ai.djl.huggingface.translator.ZeroShotClassificationTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.nlp.translator.ZeroShotClassificationInput;
import ai.djl.modality.nlp.translator.ZeroShotClassificationOutput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
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

public class ZeroShotClassificationTranslatorTest {

    private static final float[][][] ARRAYS = {
        {{-2.6497f, 0.8422f, 2.0454f}},
        {{2.9208f, 0.1431f, -3.3346f}},
        {{1.6723f, 1.0156f, -3.1728f}},
        {{-1.795f, 0.983f, 0.9644f}}
    };

    private static int index;

    @Test
    public void testZeroShotClassificationTranslator()
            throws ModelException, IOException, TranslateException {
        String text = "one day I will see the world";
        String[] candidates = {"travel", "cooking", "dancing", "exploration"};

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            NDArray arr = manager.create(ARRAYS[index++ % 4]);
                            return new NDList(arr);
                        },
                        "model");

        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);

        Criteria<ZeroShotClassificationInput, ZeroShotClassificationOutput> criteria =
                Criteria.builder()
                        .setTypes(
                                ZeroShotClassificationInput.class,
                                ZeroShotClassificationOutput.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "google-bert/bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new ZeroShotClassificationTranslatorFactory())
                        .build();

        try (ZooModel<ZeroShotClassificationInput, ZeroShotClassificationOutput> model =
                        criteria.loadModel();
                Predictor<ZeroShotClassificationInput, ZeroShotClassificationOutput> predictor =
                        model.newPredictor()) {
            ZeroShotClassificationInput input = new ZeroShotClassificationInput(text, candidates);
            ZeroShotClassificationOutput res = predictor.predict(input);
            Assertions.assertAlmostEquals(res.getScores()[1], 0.940441);
            Assert.assertEquals(res.getLabels()[1], "exploration");
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "google-bert/bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new ZeroShotClassificationTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            ZeroShotClassificationInput zsc =
                    new ZeroShotClassificationInput(text, candidates, true);
            input.add(JsonUtils.toJson(zsc));
            Output out = predictor.predict(input);
            ZeroShotClassificationOutput res =
                    (ZeroShotClassificationOutput) out.getData().getAsObject();
            Assertions.assertAlmostEquals(res.getScores()[1], 0.251434);
            Assert.assertEquals(res.getLabels()[0], "travel");
        }

        try (Model model = Model.newInstance("test")) {
            model.setBlock(block);
            Map<String, String> options = new HashMap<>();
            options.put("hasParameter", "false");
            model.load(modelDir, "test", options);

            ZeroShotClassificationTranslatorFactory factory =
                    new ZeroShotClassificationTranslatorFactory();
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
