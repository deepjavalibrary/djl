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
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.modality.nlp.generate.GPTConfig;
import ai.djl.modality.nlp.generate.LMBlock;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.training.util.ProgressBar;
import ai.djl.util.Pair;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public final class LLMBlock {

    private LLMBlock() {}

    public static int main(String[] args)
            throws ModelNotFoundException, MalformedModelException, IOException {
        mainOnnx();
        mainPt();
        return 0;
    }

    public static Pair<Block, List<Model>> getLMBlock(
            String[] modelUrls, String engine, String modelName)
            throws ModelNotFoundException, MalformedModelException, IOException {
        Block[] blocks;
        List<Model> models = new LinkedList<>();
        // modelUrl can be replaced to local model file
        blocks = new Block[modelUrls.length];
        for (int i = 0; i < modelUrls.length; i++) {
            Criteria<NDList, NDList> criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelUrls(modelUrls[i])
                            .optEngine(engine)
                            .optProgress(new ProgressBar())
                            .build();
            Model model = criteria.loadModel();
            blocks[i] = model.getBlock();
            models.add(model);
        }

        return new Pair<>(
                // Creating a LMBlock calls GPT2PtLMBlock.java which is engine specific, whose
                // package
                // `pytorch-engines.main` cannot be loaded here.
                Engine.getEngine(engine).newLMBlock(modelName, new GPTConfig(), blocks), models);
    }

    public static void mainOnnx()
            throws ModelNotFoundException, MalformedModelException, IOException {
        String[] modelUrls = {"https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2.onnx.zip"};

        Pair<Block, List<Model>> result = LLMBlock.getLMBlock(modelUrls, "OnnxRuntime", "GPT2");
        LMBlock generator = (LMBlock) result.getKey();
        List<Model> models = result.getValue();

        try (NDManager manager = NDManager.newBaseManager()) {

            /////////////////////////////////////////////
            // Inference without cached key_values input
            /////////////////////////////////////////////

            long[] inputArray = {40, 2883, 6155, 351, 616, 13779};
            int numBatch = 2;

            NDArray inputIds = manager.create(inputArray, new Shape(2, inputArray.length / 2));

            NDArray positionIds =
                    manager.arange(0, inputIds.getShape().size(1), 1, DataType.INT64)
                            .reshape(1, -1)
                            .repeat(0, numBatch);

            NDArray attentionMask = manager.ones(positionIds.getShape(), DataType.INT64);

            CausalLMOutput outInit =
                    generator.forward(
                            new NDList(inputIds, positionIds, attentionMask), null, manager);

            /////////////////////////////////////////////
            // Inference with cached key_values input
            /////////////////////////////////////////////

            long pastSeqLen = outInit.getPastKeyValuesList().get(0).getShape().size(2);
            inputIds = manager.create(new long[] {404, 403, 402, 401}, new Shape(numBatch, 2));
            positionIds =
                    manager.arange(
                                    pastSeqLen,
                                    pastSeqLen + inputIds.getShape().getLastDimension(),
                                    1,
                                    DataType.INT64)
                            .reshape(1, -1)
                            .repeat(0, numBatch);
            attentionMask =
                    manager.ones(
                                    new Shape(
                                            1, pastSeqLen + inputIds.getShape().getLastDimension()),
                                    DataType.INT64)
                            .reshape(1, -1)
                            .repeat(0, numBatch);

            generator.forward(
                    new NDList(inputIds, positionIds, attentionMask),
                    outInit.getPastKeyValuesList(),
                    manager);
        }
        models.forEach(Model::close);
    }

    public static void mainPt()
            throws ModelNotFoundException, MalformedModelException, IOException {
        String[] modelUrls = {"https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2.pt.zip"};

        Pair<Block, List<Model>> result = LLMBlock.getLMBlock(modelUrls, "PyTorch", "GPT2");
        LMBlock generator = (LMBlock) result.getKey();
        List<Model> models = result.getValue();

        try (NDManager manager = NDManager.newBaseManager()) {
            /////////////////////////////////////////////
            // Inference without cached key_values input
            /////////////////////////////////////////////

            int[] inputArray = {40, 2883, 6155, 351, 616, 13779};
            int numBatch = 2;

            NDArray inputIds = manager.create(inputArray, new Shape(2, inputArray.length / 2));

            NDArray positionIds =
                    manager.arange(0, inputIds.getShape().size(1), 1, DataType.INT64)
                            .reshape(1, -1)
                            .repeat(0, numBatch);

            NDArray attentionMask = manager.ones(positionIds.getShape());

            CausalLMOutput outInit =
                    generator.forward(
                            new NDList(inputIds, positionIds, attentionMask), null, manager);

            /////////////////////////////////////////////
            // Inference with cached key_values input
            /////////////////////////////////////////////

            long pastSeqLen = outInit.getPastKeyValuesList().get(0).getShape().size(2);
            inputIds = manager.create(new int[] {404, 403, 402, 401}, new Shape(numBatch, 2));
            positionIds =
                    manager.arange(
                                    pastSeqLen,
                                    pastSeqLen + inputIds.getShape().getLastDimension(),
                                    1,
                                    DataType.INT64)
                            .reshape(1, -1)
                            .repeat(0, numBatch);
            attentionMask =
                    manager.ones(new Shape(1, pastSeqLen + inputIds.getShape().getLastDimension()))
                            .reshape(1, -1)
                            .repeat(0, numBatch);

            generator.forward(
                    new NDList(inputIds, positionIds, attentionMask),
                    outInit.getPastKeyValuesList(),
                    manager);
        }
        models.forEach(Model::close);
    }
}
