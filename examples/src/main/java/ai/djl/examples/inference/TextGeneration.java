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
package ai.djl.examples.inference;

import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.CausalLMOutput;
import ai.djl.translate.LMAdapter;

import java.io.IOException;

public final class TextGeneration {

    private TextGeneration() {}

    public static void main(String[] args) {
        mainOnnx(args);
        mainPt(args);
    }

    public static void mainOnnx(String[] args) {
        String[] modelUrls =
                new String[] {
                    "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/gpt2_onnx/decoder_model_merged.onnx"
                };

        try (LMAdapter generator =
                        Engine.getEngine("OnnxRuntime").newLMAdapter("GPT2", modelUrls);
             NDManager manager = NDManager.newBaseManager()) {

            /////////////////////////////////////////////
            // Inference without cached key_values input
            /////////////////////////////////////////////

            long[] inputArray = {40, 2883, 6155, 351, 616, 13779};
            int numBatch = 2;

            NDArray inputIds = manager.create(inputArray, new Shape(2, inputArray.length / 2));

            NDArray positionIds =
                    manager.arange(0, inputIds.getShape().size(-1), 1, DataType.INT64)
                            .reshape(1, -1)
                            .repeat(0, numBatch);

            NDArray attentionMask = manager.ones(positionIds.getShape(), DataType.INT64);

            CausalLMOutput outInit =
                    generator.forward(
                            new NDList(inputIds, positionIds, attentionMask), null, manager);

            /////////////////////////////////////////////
            // Inference with cached key_values input
            /////////////////////////////////////////////

            long pastSeqLen = outInit.pastKeyValuesList.get(0).getShape().size(-2);
            inputIds = manager.create(new long[] {404, 403, 402, 401}, new Shape(numBatch, 2));
            positionIds =
                    manager.arange(
                                    pastSeqLen,
                                    pastSeqLen + inputIds.getShape().get(-1),
                                    1,
                                    DataType.INT64)
                            .reshape(1, -1)
                            .repeat(0, numBatch);
            attentionMask =
                    manager.ones(
                                    new Shape(1, pastSeqLen + inputIds.getShape().get(-1)),
                                    DataType.INT64)
                            .reshape(1, -1)
                            .repeat(0, numBatch);

            CausalLMOutput out =
                    generator.forward(
                            new NDList(inputIds, positionIds, attentionMask),
                            outInit.pastKeyValuesList,
                            manager);

            System.out.println(out.logits);
            System.out.println(out.pastKeyValuesList);

        } catch (ModelNotFoundException | MalformedModelException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void mainPt(String[] args) {
        String[] modelUrls = {
            "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2_init.pt",
            "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2.pt"
        };

        try (LMAdapter generator = Engine.getEngine("PyTorch").newLMAdapter("GPT2", modelUrls);
             NDManager manager = NDManager.newBaseManager()) {

            /////////////////////////////////////////////
            // Inference without cached key_values input
            /////////////////////////////////////////////

            int[] inputArray = {40, 2883, 6155, 351, 616, 13779};
            int numBatch = 2;

            NDArray inputIds = manager.create(inputArray, new Shape(2, inputArray.length / 2));

            NDArray positionIds =
                    manager.arange(0, inputIds.getShape().size(-1), 1, DataType.INT64)
                            .reshape(1, -1)
                            .repeat(0, numBatch);

            NDArray attentionMask = manager.ones(positionIds.getShape());

            CausalLMOutput outInit =
                    generator.forward(
                            new NDList(inputIds, positionIds, attentionMask), null, manager);

            /////////////////////////////////////////////
            // Inference with cached key_values input
            /////////////////////////////////////////////

            long pastSeqLen = outInit.pastKeyValuesList.get(0).getShape().size(-2);
            inputIds = manager.create(new int[] {404, 403, 402, 401}, new Shape(numBatch, 2));
            positionIds =
                    manager.arange(
                                    pastSeqLen,
                                    pastSeqLen + inputIds.getShape().get(-1),
                                    1,
                                    DataType.INT64)
                            .reshape(1, -1)
                            .repeat(0, numBatch);
            attentionMask =
                    manager.ones(new Shape(1, pastSeqLen + inputIds.getShape().get(-1)))
                            .reshape(1, -1)
                            .repeat(0, numBatch);

            CausalLMOutput out =
                    generator.forward(
                            new NDList(inputIds, positionIds, attentionMask),
                            outInit.pastKeyValuesList,
                            manager);

            System.out.println(out.logits);
            System.out.println(out.pastKeyValuesList);

        } catch (ModelNotFoundException | MalformedModelException | IOException e) {
            throw new RuntimeException(e);
        }
    }
}
