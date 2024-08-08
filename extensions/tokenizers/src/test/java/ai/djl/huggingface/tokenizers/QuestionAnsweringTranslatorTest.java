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
import ai.djl.huggingface.translator.QuestionAnsweringTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
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

public class QuestionAnsweringTranslatorTest {

    @Test
    public void testQATranslator() throws ModelException, IOException, TranslateException {
        String question = "When did BBC Japan start broadcasting?";
        String paragraph =
                "BBC Japan was a general entertainment Channel. "
                        + "Which operated between December 2004 and April 2006. "
                        + "It ceased operations after its Japanese distributor folded.";

        Block block =
                new LambdaBlock(
                        a -> {
                            NDManager manager = a.getManager();
                            long[][] start = new long[1][36];
                            long[][] end = new long[1][36];
                            start[0][0] = 2;
                            start[0][21] = 1;
                            end[0][0] = 2;
                            end[0][20] = 1;
                            NDArray arr1 = manager.create(start);
                            NDArray arr2 = manager.create(end);
                            return new NDList(arr1, arr2);
                        },
                        "model");
        Path modelDir = Paths.get("build/model");
        Files.createDirectories(modelDir);

        Criteria<QAInput, String> criteria =
                Criteria.builder()
                        .setTypes(QAInput.class, String.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-cased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new QuestionAnsweringTranslatorFactory())
                        .build();

        try (ZooModel<QAInput, String> model = criteria.loadModel();
                Predictor<QAInput, String> predictor = model.newPredictor()) {
            QAInput input = new QAInput(question, paragraph);
            String res = predictor.predict(input);
            Assert.assertEquals(res, "December 2004");
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optEngine("PyTorch")
                        .optArgument("tokenizer", "bert-base-cased")
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new QuestionAnsweringTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add("question", question);
            input.add("paragraph", paragraph);
            Output res = predictor.predict(input);
            Assert.assertEquals(res.getAsString(0), "December 2004");

            Assert.assertThrows(
                    "Input data is empty.",
                    TranslateException.class,
                    () -> predictor.predict(new Input()));

            Assert.assertThrows(
                    "Missing question or context in input.",
                    TranslateException.class,
                    () -> {
                        Input req = new Input();
                        req.add("something", "false");
                        predictor.predict(req);
                    });

            Assert.assertThrows(
                    "Input is not a valid json.",
                    TranslateException.class,
                    () -> {
                        Input req = new Input();
                        req.addProperty("Content-Type", "application/json; charset=utf-8");
                        req.add("Invalid json");
                        predictor.predict(req);
                    });

            Assert.assertThrows(
                    "Missing question or context in json.",
                    TranslateException.class,
                    () -> {
                        Input req = new Input();
                        req.addProperty("Content-Type", "application/json");
                        req.add(JsonUtils.GSON.toJson(new QAInput(question, null)));
                        predictor.predict(req);
                    });
        }

        try (Model model = Model.newInstance("test")) {
            model.setBlock(block);
            Map<String, String> options = new HashMap<>();
            options.put("hasParameter", "false");
            model.load(modelDir, "test", options);

            QuestionAnsweringTranslatorFactory factory = new QuestionAnsweringTranslatorFactory();
            Map<String, String> arguments = new HashMap<>();

            Assert.assertThrows(
                    TranslateException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));

            arguments.put("tokenizer", "bert-base-cased");

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(String.class, Integer.class, model, arguments));
        }
    }
}
