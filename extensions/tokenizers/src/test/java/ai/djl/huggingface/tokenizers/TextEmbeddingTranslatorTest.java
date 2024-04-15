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
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TextEmbeddingTranslatorTest {

    @Test
    public void testTextEmbeddingTranslator()
            throws ModelException, IOException, TranslateException {
        String text = "This is an example sentence";

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            NDArray arr = manager.ones(new Shape(1, 7, 384));
                            arr.setName("last_hidden_state");
                            return new NDList(arr);
                        },
                        "model");
        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);

        Criteria<String, float[]> criteria =
                Criteria.builder()
                        .setTypes(String.class, float[].class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<String, float[]> model = criteria.loadModel();
                Predictor<String, float[]> predictor = model.newPredictor()) {
            float[] res = predictor.predict(text);
            Assert.assertEquals(res.length, 384);
            Assertions.assertAlmostEquals(res[0], 0.05103);
        }

        // pooling_mode_max_tokens
        criteria =
                Criteria.builder()
                        .setTypes(String.class, float[].class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optArgument("pooling", "max")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<String, float[]> model = criteria.loadModel();
                Predictor<String, float[]> predictor = model.newPredictor()) {
            float[] res = predictor.predict(text);
            Assert.assertEquals(res.length, 384);
            Assertions.assertAlmostEquals(res[0], 1.0);
        }

        // pooling_mean_sqrt_len_tokens
        criteria =
                Criteria.builder()
                        .setTypes(String.class, float[].class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optArgument("pooling", "mean_sqrt_len")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<String, float[]> model = criteria.loadModel();
                Predictor<String, float[]> predictor = model.newPredictor()) {
            float[] res = predictor.predict(text);
            Assert.assertEquals(res.length, 384);
            Assertions.assertAlmostEquals(res[0], 0.05103104);
        }

        // pooling_weightedmean_tokens
        criteria =
                Criteria.builder()
                        .setTypes(String.class, float[].class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optArgument("pooling", "weightedmean")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<String, float[]> model = criteria.loadModel();
                Predictor<String, float[]> predictor = model.newPredictor()) {
            float[] res = predictor.predict(text);
            Assert.assertEquals(res.length, 384);
            Assertions.assertAlmostEquals(res[0], 0.05103104);
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optArgument("pooling", "cls")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add(text);
            Output out = predictor.predict(input);
            float[] res = JsonUtils.GSON.fromJson(out.getAsString(0), float[].class);
            Assert.assertEquals(res.length, 384);
            Assertions.assertAlmostEquals(res[0], 0.05103);

            input = new Input();
            Map<String, String> map = new HashMap<>();
            map.put("inputs", text);
            input.add(JsonUtils.GSON.toJson(map));
            input.addProperty("Content-Type", "application/json");
            out = predictor.predict(input);
            res = (float[]) out.getData().getAsObject();
            Assert.assertEquals(res.length, 384);
            Assertions.assertAlmostEquals(res[0], 0.05103);
        }

        try (Model model = Model.newInstance("test")) {
            model.setBlock(block);
            Map<String, String> options = new HashMap<>();
            options.put("hasParameter", "false");
            model.load(modelDir, "test", options);

            TextEmbeddingTranslatorFactory factory = new TextEmbeddingTranslatorFactory();
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

    @Test
    public void testTextEmbeddingBatchTranslator()
            throws ModelException, IOException, TranslateException {
        String[] text = {"This is an example sentence", "This is the second sentence"};

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            NDArray arr = manager.ones(new Shape(2, 7, 384));
                            arr.setName("last_hidden_state");
                            return new NDList(arr);
                        },
                        "model");
        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);

        Criteria<String[], float[][]> criteria =
                Criteria.builder()
                        .setTypes(String[].class, float[][].class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optArgument("padding", "true")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<String[], float[][]> model = criteria.loadModel();
                Predictor<String[], float[][]> predictor = model.newPredictor()) {
            float[][] res = predictor.predict(text);
            Assert.assertEquals(res[0].length, 384);
            Assertions.assertAlmostEquals(res[0][0], 0.05103);
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optArgument("padding", "true")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add(JsonUtils.GSON.toJson(text));
            input.addProperty("Content-Type", "application/json");
            Output out = predictor.predict(input);
            float[][] res = (float[][]) out.getData().getAsObject();
            Assert.assertEquals(res[0].length, 384);
            Assertions.assertAlmostEquals(res[0][0], 0.05103);

            input = new Input();
            Map<String, String[]> map = new HashMap<>();
            map.put("inputs", text);
            input.add(JsonUtils.GSON.toJson(map));
            input.addProperty("Content-Type", "application/json");
            out = predictor.predict(input);
            res = (float[][]) out.getData().getAsObject();
            Assert.assertEquals(res[0].length, 384);
            Assertions.assertAlmostEquals(res[0][0], 0.05103);

            Assert.assertThrows(
                    () -> {
                        Input empty = new Input();
                        empty.add(JsonUtils.GSON.toJson(new HashMap<>()));
                        empty.addProperty("Content-Type", "application/json");
                        predictor.predict(empty);
                    });

            Assert.assertThrows(
                    () -> {
                        Input empty = new Input();
                        empty.add("{ \"invalid json\"");
                        empty.addProperty("Content-Type", "application/json");
                        predictor.predict(empty);
                    });
        }
    }

    @Test
    public void testTextEmbeddingTranslatorServingBatch()
            throws ModelException, IOException, TranslateException {
        String[] text = {"This is an example sentence", "This is the second sentence"};

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            NDArray arr = manager.ones(new Shape(4, 7, 384));
                            arr.setName("last_hidden_state");
                            return new NDList(arr);
                        },
                        "model");
        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);

        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input1 = new Input();
            input1.add(JsonUtils.GSON.toJson(text));
            input1.addProperty("Content-Type", "application/json");

            Input input2 = new Input();
            Map<String, String[]> map = new HashMap<>();
            map.put("inputs", text);
            input2.add(JsonUtils.GSON.toJson(map));
            input2.addProperty("Content-Type", "application/json");
            List<Input> batchInput = Arrays.asList(input1, input2);

            List<Output> batchOutput = predictor.batchPredict(batchInput);
            Assert.assertEquals(batchOutput.size(), 2);
            float[][] res = (float[][]) batchOutput.get(0).getData().getAsObject();
            Assert.assertEquals(res[0].length, 384);
            Assertions.assertAlmostEquals(res[0][0], 0.05103);
        }
    }
}
