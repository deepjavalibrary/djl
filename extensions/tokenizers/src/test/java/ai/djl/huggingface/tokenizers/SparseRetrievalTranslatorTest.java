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

import ai.djl.ModelException;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.nlp.EmbeddingOutput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.Assertions;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;

import com.google.gson.JsonArray;

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

public class SparseRetrievalTranslatorTest {

    @Test
    public void testSparseRetrievalTranslator()
            throws ModelException, IOException, TranslateException {
        String[] text = {"This is an example sentence", "What is sparse retrieval?"};

        Path modelDir = Paths.get("build/model");
        Path sparseLinear = modelDir.resolve("sparse_linear.safetensors");
        Files.createDirectories(modelDir);
        try (NDManager manager = NDManager.newBaseManager("PyTorch")) {
            NDArray weight = manager.ones(new Shape(1, 1024));
            weight.setName("weight");
            NDArray bias = manager.ones(new Shape(1));
            bias.setName("bias");
            NDList linear = new NDList(weight, bias);
            try (OutputStream os = Files.newOutputStream(sparseLinear)) {
                linear.encode(os, NDList.Encoding.SAFETENSORS);
            }
        }

        Criteria<String, EmbeddingOutput> criteria =
                Criteria.builder()
                        .setTypes(String.class, EmbeddingOutput.class)
                        .optModelPath(modelDir)
                        .optEngine("PyTorch")
                        .optArgument("blockFactory", "ai.djl.nn.OnesBlockFactory")
                        .optArgument("block_shapes", "(2,9,1024)")
                        .optArgument("block_names", "last_hidden_state")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optArgument("sparse", true)
                        .optArgument("returnDenseEmbedding", true)
                        .optArgument("pooling", "cls")
                        .optArgument("sparseLinear", sparseLinear.toAbsolutePath().toString())
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<String, EmbeddingOutput> model = criteria.loadModel();
                Predictor<String, EmbeddingOutput> predictor = model.newPredictor()) {
            EmbeddingOutput res = predictor.predict(text[0]);
            Map<String, Float> tokenWeights = res.getLexicalWeights();
            Assert.assertNotNull(res.getDenseEmbedding());
            Assert.assertEquals(tokenWeights.size(), 5);
            Assertions.assertAlmostEquals(tokenWeights.get("2023"), 1025.0);
        }

        Criteria<Input, Output> criteria2 =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optArgument("blockFactory", "ai.djl.nn.OnesBlockFactory")
                        .optArgument("block_shapes", "(2,9,1024)")
                        .optArgument("block_names", "last_hidden_state")
                        .optArgument("tokenizer", "bert-base-uncased")
                        .optArgument("sparse", true)
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                        .build();

        try (ZooModel<Input, Output> model = criteria2.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add(text[0]);
            Output out = predictor.predict(input);
            EmbeddingOutput res =
                    JsonUtils.GSON.fromJson(out.getAsString(0), EmbeddingOutput.class);
            Map<String, Float> tokenWeights = res.getLexicalWeights();
            Assert.assertNull(res.getDenseEmbedding());
            Assert.assertEquals(tokenWeights.size(), 5);
            Assertions.assertAlmostEquals(tokenWeights.get("2023"), 1025.0);

            // batch serving predict
            Input input2 = new Input();
            input2.add(text[1]);
            List<Input> batchInput = Arrays.asList(input, input2);
            List<Output> batchOutput = predictor.batchPredict(batchInput);
            out = batchOutput.get(0);
            res = JsonUtils.GSON.fromJson(out.getAsString(0), EmbeddingOutput.class);
            tokenWeights = res.getLexicalWeights();
            Assert.assertNull(res.getDenseEmbedding());
            Assert.assertEquals(tokenWeights.size(), 5);
            Assertions.assertAlmostEquals(tokenWeights.get("2023"), 1025.0);

            // client side batch
            input = new Input();
            Map<String, String[]> map = new HashMap<>();
            map.put("inputs", text);
            input.add(JsonUtils.GSON.toJson(map));
            input.addProperty("Content-Type", "application/json; charset=utf-8");
            out = predictor.predict(input);
            JsonArray batch = JsonUtils.GSON.fromJson(out.getAsString(0), JsonArray.class);
            res = JsonUtils.GSON.fromJson(batch.get(0), EmbeddingOutput.class);
            tokenWeights = res.getLexicalWeights();
            Assert.assertNull(res.getDenseEmbedding());
            Assert.assertEquals(tokenWeights.size(), 5);
            Assertions.assertAlmostEquals(tokenWeights.get("2023"), 1025.0);
        }
    }
}
