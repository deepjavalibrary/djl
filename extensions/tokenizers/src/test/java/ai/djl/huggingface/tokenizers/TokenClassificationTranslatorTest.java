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
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class TokenClassificationTranslatorTest {

    @BeforeClass
    public void setUp() throws IOException {
        Path modelDir = Paths.get("build/token_classification");
        Files.createDirectories(modelDir);
        Path path = modelDir.resolve("config.json");
        Map<String, Map<String, String>> map = new HashMap<>();
        Map<String, String> id2label = new HashMap<>();
        id2label.put("0", "O");
        id2label.put("1", "I-PER");
        id2label.put("2", "ORG");
        id2label.put("3", "LOC");
        id2label.put("4", "MISC");
        map.put("id2label", id2label);
        try (Writer writer = Files.newBufferedWriter(path)) {
            writer.write(JsonUtils.GSON.toJson(map));
        }
    }

    @AfterClass
    public void tierDown() {
        Utils.deleteQuietly(Paths.get("build/token_classification"));
    }

    @Test
    public void testTokenClassificationTranslator()
            throws ModelException, IOException, TranslateException {
        String text =
                "Apple was founded in 1976 by Steve Jobs, Steve Wozniak and Ronald sell Apple I"
                        + " personal computer.";

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            float[][] logits = new float[24][5];
                            logits[1][2] = 1f;
                            logits[7][1] = 1f;
                            logits[8][1] = 1f;
                            logits[10][1] = 1f;
                            logits[11][1] = 1f;
                            logits[12][1] = 1f;
                            logits[13][1] = 1f;
                            logits[14][1] = 1f;
                            logits[16][1] = 1f;
                            logits[17][1] = 1f;
                            logits[18][4] = 1f;
                            logits[19][4] = 1f;
                            NDArray arr = manager.create(logits);
                            arr = arr.expandDims(0);
                            return new NDList(arr);
                        },
                        "model");

        Path modelDir = Paths.get("build/token_classification");
        Criteria<String, NamedEntity[]> criteria =
                Criteria.builder()
                        .setTypes(String.class, NamedEntity[].class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "FacebookAI/roberta-base")
                        .optArgument("softmax", true)
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TokenClassificationTranslatorFactory())
                        .build();

        try (ZooModel<String, NamedEntity[]> model = criteria.loadModel();
                Predictor<String, NamedEntity[]> predictor = model.newPredictor()) {
            NamedEntity[] res = predictor.predict(text);
            Assert.assertEquals(res.length, 12);
            Assert.assertEquals(res[0].getEntity(), "ORG");
            Assert.assertEquals(res[0].getIndex(), 1);
            Assert.assertEquals(res[0].getWord(), "Apple");
            Assert.assertEquals(res[0].getStart(), 0);
            Assert.assertEquals(res[0].getEnd(), 5);
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "FacebookAI/roberta-base")
                        .optArgument("softmax", true)
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TokenClassificationTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add(text);
            Output out = predictor.predict(input);
            NamedEntity[] res = (NamedEntity[]) out.getData().getAsObject();
            Assert.assertEquals(res[1].getEntity(), "I-PER");
            Assert.assertEquals(res[1].getStart(), 29);
            Assert.assertEquals(res[1].getEnd(), 34);
        }

        // simple aggregation
        Criteria<String, NamedEntity[]> criteria3 =
                criteria.toBuilder().optArgument("aggregation_strategy", "simple").build();
        try (ZooModel<String, NamedEntity[]> model = criteria3.loadModel();
                Predictor<String, NamedEntity[]> predictor = model.newPredictor()) {
            NamedEntity[] res = predictor.predict(text);
            Assert.assertEquals(res.length, 5);
            Assert.assertEquals(res[1].getEntity(), "PER");
            Assert.assertEquals(res[1].getWord(), " Steve Jobs");
            Assert.assertEquals(res[1].getStart(), 29);
            Assert.assertEquals(res[1].getEnd(), 39);
        }

        Criteria<String, NamedEntity[]> criteria4 =
                criteria.toBuilder().optArgument("aggregation_strategy", "first").build();
        try (ZooModel<String, NamedEntity[]> model = criteria3.loadModel();
                Predictor<String, NamedEntity[]> predictor = model.newPredictor()) {
            NamedEntity[] res = predictor.predict(text);
            Assert.assertEquals(res.length, 4);
            Assert.assertEquals(res[1].getEntity(), "PER");
            Assert.assertEquals(res[1].getWord(), " Steve Jobs, Steve Wozniak");
            Assert.assertEquals(res[1].getStart(), 29);
            Assert.assertEquals(res[1].getEnd(), 54);
        }

        Criteria<String, NamedEntity[]> criteria5 =
                criteria.toBuilder().optArgument("aggregation_strategy", "max").build();
        try (ZooModel<String, NamedEntity[]> model = criteria3.loadModel();
                Predictor<String, NamedEntity[]> predictor = model.newPredictor()) {
            NamedEntity[] res = predictor.predict(text);
            Assert.assertEquals(res.length, 4);
            Assert.assertEquals(res[1].getEntity(), "PER");
            Assert.assertEquals(res[1].getWord(), " Steve Jobs, Steve Wozniak");
            Assert.assertEquals(res[1].getStart(), 29);
            Assert.assertEquals(res[1].getEnd(), 54);
        }

        Criteria<String, NamedEntity[]> criteria6 =
                criteria.toBuilder().optArgument("aggregation_strategy", "average").build();
        try (ZooModel<String, NamedEntity[]> model = criteria3.loadModel();
                Predictor<String, NamedEntity[]> predictor = model.newPredictor()) {
            NamedEntity[] res = predictor.predict(text);
            Assert.assertEquals(res.length, 4);
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

            arguments.put("tokenizer", "FacebookAI/roberta-base");

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));
        }
    }
}
