/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.huggingface.translator.TokenClassificationTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.nlp.translator.NamedEntity;
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
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class TokenClassificationTranslatorTest {

    @AfterClass
    public void tierDown() {
        Utils.deleteQuietly(Paths.get("build/token_classification"));
    }

    @Test
    public void testTokenClassificationTranslator()
            throws ModelException, IOException, TranslateException {
        String text = "My name is Wolfgang and I live in Berlin.";

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            float[][] logits = new float[12][9];
                            logits[4][3] = 1;
                            logits[9][7] = 1;
                            NDArray arr = manager.create(logits);
                            arr = arr.expandDims(0);
                            return new NDList(arr);
                        },
                        "model");
        Path modelDir = Paths.get("build/token_classification");
        Files.createDirectories(modelDir);
        Path path = modelDir.resolve("config.json");
        Map<String, Map<String, String>> map = new HashMap<>();
        Map<String, String> id2label = new HashMap<>();
        id2label.put("0", "O");
        id2label.put("3", "B-PER");
        id2label.put("7", "B-LOC");
        map.put("id2label", id2label);
        try (Writer writer = Files.newBufferedWriter(path)) {
            writer.write(JsonUtils.GSON.toJson(map));
        }

        Criteria<String, NamedEntity[]> criteria =
                Criteria.builder()
                        .setTypes(String.class, NamedEntity[].class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TokenClassificationTranslatorFactory())
                        .build();

        try (ZooModel<String, NamedEntity[]> model = criteria.loadModel();
                Predictor<String, NamedEntity[]> predictor = model.newPredictor()) {
            NamedEntity[] res = predictor.predict(text);
            Assert.assertEquals(res[0].getEntity(), "B-PER");
            Assertions.assertAlmostEquals(res[0].getScore(), 0.2536117);
            Assert.assertEquals(res[0].getIndex(), 4);
            Assert.assertEquals(res[0].getWord(), "wolfgang");
            Assert.assertEquals(res[0].getStart(), 11);
            Assert.assertEquals(res[0].getEnd(), 19);
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TokenClassificationTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add(text);
            Output out = predictor.predict(input);
            NamedEntity[] res = (NamedEntity[]) out.getData().getAsObject();
            Assert.assertEquals(res[0].getEntity(), "B-PER");
        }

        try (Model model = Model.newInstance("test")) {
            model.setBlock(block);
            Map<String, String> options = new HashMap<>();
            options.put("hasParameter", "false");
            model.load(modelDir, "test", options);

            TokenClassificationTranslatorFactory factory =
                    new TokenClassificationTranslatorFactory();
            Map<String, String> arguments = new HashMap<>();

            Assert.assertThrows(
                    TranslateException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));

            arguments.put("tokenizer", "bert-base-uncased");

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));
        }
    }
}
