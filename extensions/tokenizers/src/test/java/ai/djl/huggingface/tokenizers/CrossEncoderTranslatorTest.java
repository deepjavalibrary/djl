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
import ai.djl.huggingface.translator.TextClassificationTranslatorFactory;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.StringPair;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CrossEncoderTranslatorTest {

    @Test
    @SuppressWarnings("unchecked")
    public void testCrossEncoderTranslator()
            throws ModelException, IOException, TranslateException {
        String text1 = "Sentence 1";
        String text2 = "Sentence 2";
        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            NDArray array = manager.create(new float[] {-0.7329f});
                            return new NDList(array);
                        },
                        "model");
        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);

        Criteria<StringPair, float[]> criteria =
                Criteria.builder()
                        .setTypes(StringPair.class, float[].class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-cased")
                        .optArgument("tokenizerPath", modelDir)
                        .optArgument("reranking", true)
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<StringPair, float[]> model = criteria.loadModel();
                Predictor<StringPair, float[]> predictor = model.newPredictor()) {
            StringPair input = new StringPair(text1, text2);
            float[] res = predictor.predict(input);
            Assert.assertEquals(res[0], 0.32456556f, 0.0001);
        }

        criteria =
                Criteria.builder()
                        .setTypes(StringPair.class, float[].class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-cased")
                        .optArgument("tokenizerPath", modelDir)
                        .optArgument("reranking", true)
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextClassificationTranslatorFactory())
                        .build();

        try (ZooModel<StringPair, float[]> model = criteria.loadModel();
                Predictor<StringPair, float[]> predictor = model.newPredictor()) {
            StringPair input = new StringPair(text1, text2);
            float[] res = predictor.predict(input);
            Assert.assertEquals(res[0], 0.32456556f, 0.0001);
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-cased")
                        .optArgument("reranking", true)
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add("key", text1);
            input.add("value", text2);
            Output res = predictor.predict(input);
            float[] buf = (float[]) res.getData().getAsObject();
            Assert.assertEquals(buf[0], 0.32455865, 0.0001);

            input = new Input();
            input.add("text", text1);
            input.add("text_pair", text2);
            res = predictor.predict(input);
            buf = (float[]) res.getData().getAsObject();
            Assert.assertEquals(buf[0], 0.32455865, 0.0001);

            input = new Input();
            input.addProperty("Content-Type", "application/json; charset=utf-8");
            input.add("data", "{\"text\": \"" + text1 + "\", \"text_pair\": \"" + text2 + "\"}");
            res = predictor.predict(input);
            buf = (float[]) res.getData().getAsObject();
            Assert.assertEquals(buf[0], 0.32455865, 0.0001);

            input = new Input();
            input.addProperty("Content-Type", "application/json; charset=utf-8");
            input.add("data", "{\"key\": \"" + text1 + "\", \"value\": \"" + text2 + "\"}");
            res = predictor.predict(input);
            buf = (float[]) res.getData().getAsObject();
            Assert.assertEquals(buf[0], 0.32455865, 0.0001);

            input = new Input();
            input.addProperty("Content-Type", "application/json");
            input.add("data", "{\"query\": \"" + text1 + "\", \"texts\": [\"" + text2 + "\"]}");
            res = predictor.predict(input);
            buf = ((List<float[]>) res.getData().getAsObject()).get(0);
            Assert.assertEquals(buf[0], 0.32455865, 0.0001);

            input = new Input();
            input.addProperty("Content-Type", "application/json");
            input.add("data", "{\"query\": \"" + text1 + "\", \"texts\": [\"" + text2 + "\"]}");
            res = predictor.predict(input);
            buf = ((List<float[]>) res.getData().getAsObject()).get(0);
            Assert.assertEquals(buf[0], 0.32455865, 0.0001);

            input = new Input();
            input.addProperty("Content-Type", "application/json");
            input.add("data", "[{\"text\": \"" + text1 + "\", \"text_pair\": \"" + text2 + "\"}]");
            res = predictor.predict(input);
            buf = ((List<float[]>) res.getData().getAsObject()).get(0);
            Assert.assertEquals(buf[0], 0.32455865, 0.0001);

            Assert.assertThrows(TranslateException.class, () -> predictor.predict(new Input()));

            Assert.assertThrows(
                    TranslateException.class,
                    () -> {
                        Input req = new Input();
                        req.add("something", "false");
                        predictor.predict(req);
                    });

            Assert.assertThrows(
                    TranslateException.class,
                    () -> {
                        Input req = new Input();
                        req.addProperty("Content-Type", "application/json");
                        req.add("Invalid json");
                        predictor.predict(req);
                    });

            Assert.assertThrows(
                    TranslateException.class,
                    () -> {
                        Input req = new Input();
                        req.addProperty("Content-Type", "application/json");
                        req.add(JsonUtils.GSON.toJson(new StringPair(text1, null)));
                        predictor.predict(req);
                    });
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

            arguments.put("tokenizer", "bert-base-cased");
            arguments.put("reranking", "true");

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));
        }
    }
}
