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
package ai.djl.pytorch.zoo.nlp.textgeneration;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class TextGenerationTest {

    @Test
    public void testGpt2() throws TranslateException, ModelException, IOException {
        Block block =
                new LambdaBlock(
                        a -> {
                            NDList list = new NDList(25);
                            NDManager manager = a.getManager();
                            long[][] logits = new long[4][50257];
                            logits[3][257] = 1;
                            NDArray arr = manager.create(logits).expandDims(0);
                            list.add(arr);

                            for (int i = 0; i < 12 * 2; ++i) {
                                NDArray array = manager.zeros(new Shape(1, 12, 1, 64));
                                list.add(array);
                            }
                            return list;
                        },
                        "model");

        Path modelDir = Paths.get("build/text_generation");
        Files.createDirectories(modelDir);

        Criteria<NDList, CausalLMOutput> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, CausalLMOutput.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optOption("hasParameter", "false")
                        .optTranslatorFactory(new PtGptTranslatorFactory())
                        .build();

        try (ZooModel<NDList, CausalLMOutput> model = criteria.loadModel();
                Predictor<NDList, CausalLMOutput> predictor = model.newPredictor();
                NDManager manager = NDManager.newBaseManager()) {
            long[][] inputIds = {{29744, 28478, 5834, 318}};
            int len = inputIds[0].length;
            NDArray input = manager.create(inputIds);
            NDArray attentionMask = manager.ones(new Shape(1, len), DataType.INT64);
            NDArray positionIds = manager.arange(0, len, 1, DataType.INT64).expandDims(0);
            CausalLMOutput res = predictor.predict(new NDList(input, attentionMask, positionIds));
            NDArray logits = res.getLogits();
            long nextTokenId = logits.get(":, -1, :").argMax().getLong();
            Assert.assertEquals(nextTokenId, 257);
            NDList list = res.getPastKeyValuesList();
            Assert.assertEquals(list.size(), 24);
            Assert.assertEquals(res.getHiddenState().getShape().get(0), 1);
        }
    }
}
