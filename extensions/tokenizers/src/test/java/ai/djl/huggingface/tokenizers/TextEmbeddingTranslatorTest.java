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
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.LambdaBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.Assertions;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.PairList;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.OutputStream;
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

        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);
        try (NDManager manager = NDManager.newBaseManager("Rust")) {
            NDArray weight = manager.ones(new Shape(256, 384));
            weight.setName("linear.weight");
            NDList linear = new NDList(weight);
            Path file = modelDir.resolve("linear.safetensors");
            try (OutputStream os = Files.newOutputStream(file)) {
                linear.encode(os, NDList.Encoding.SAFETENSORS);
            }
            NDArray normWeight = manager.ones(new Shape(256));
            normWeight.setName("norm.weight");
            NDArray bias = manager.ones(new Shape(256));
            bias.setName("norm.bias");
            NDList norm = new NDList(normWeight, bias);
            file = modelDir.resolve("norm.safetensors");
            try (OutputStream os = Files.newOutputStream(file)) {
                norm.encode(os, NDList.Encoding.SAFETENSORS);
            }
        }

        Criteria<String, float[]> criteria =
                Criteria.builder()
                        .setTypes(String.class, float[].class)
                        .optModelPath(modelDir)
                        .optEngine("PyTorch")
                        .optArgument("blockFactory", "ai.djl.nn.OnesBlockFactory")
                        .optArgument("block_shapes", "(1,7,384)")
                        .optArgument("block_names", "last_hidden_state")
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
                        .optArgument("blockFactory", "ai.djl.nn.OnesBlockFactory")
                        .optArgument("block_shapes", "(1,7,384)")
                        .optArgument("block_names", "last_hidden_state")
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
            Assertions.assertAlmostEquals(res[0], 0.05103);
        }

        // pooling_mean_sqrt_len_tokens
        criteria =
                Criteria.builder()
                        .setTypes(String.class, float[].class)
                        .optModelPath(modelDir)
                        .optArgument("blockFactory", "ai.djl.nn.OnesBlockFactory")
                        .optArgument("block_shapes", "(1,7,384)")
                        .optArgument("block_names", "last_hidden_state")
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
                        .optArgument("blockFactory", "ai.djl.nn.OnesBlockFactory")
                        .optArgument("block_shapes", "(1,7,384)")
                        .optArgument("block_names", "last_hidden_state")
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

        // dense and layerNorm
        criteria =
                Criteria.builder()
                        .setTypes(String.class, float[].class)
                        .optModelPath(modelDir)
                        .optArgument("blockFactory", "ai.djl.nn.OnesBlockFactory")
                        .optArgument("block_shapes", "(1,7,384)")
                        .optArgument("block_names", "last_hidden_state")
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optArgument("dense", "linear.safetensors")
                        .optArgument("denseActivation", "Tanh")
                        .optArgument("layerNorm", "norm.safetensors")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<String, float[]> model = criteria.loadModel();
                Predictor<String, float[]> predictor = model.newPredictor()) {
            float[] res = predictor.predict(text);
            Assert.assertEquals(res.length, 256);
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optArgument("blockFactory", "ai.djl.nn.OnesBlockFactory")
                        .optArgument("block_shapes", "(1,7,384)")
                        .optArgument("block_names", "last_hidden_state")
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
            PairList<DataType, Shape> pairs = new PairList<>();
            pairs.add(DataType.FLOAT32, new Shape(1, 7, 384));
            model.setBlock(Blocks.onesBlock(pairs, Utils.EMPTY_ARRAY));
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
