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
package ai.djl.pytorch.engine;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.pytorch.jni.IValue;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.CausalLMOutput;
import ai.djl.translate.StepGenerator;
import ai.djl.util.NativeResource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PtStepGenerator implements StepGenerator {
    Block[] blocks;
    List<ZooModel<NDList, NDList>> models;

    public PtStepGenerator(String[] modelUrls)
            throws ModelNotFoundException, MalformedModelException, IOException {
        blocks = new Block[modelUrls.length];
        models = new ArrayList<>();
        for (int i = 0; i < modelUrls.length; i++) {
            Criteria<NDList, NDList> criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelUrls(modelUrls[i])
                            .optProgress(new ProgressBar())
                            .optEngine("PyTorch")
                            .optOption("trainParam", String.valueOf(false))
                            .build();
            ZooModel<NDList, NDList> model = criteria.loadModel();
            blocks[i] = model.getBlock();
            models.add(model);
        }
    }

    /** {@inheritDoc} */
    @Override
    public CausalLMOutput stepGeneration(
            NDList input, NativeResource<Long> pastKeyValues, NDManager manager) {
        IValue[] inputNative =
                input.stream()
                        .map(object -> IValue.from((PtNDArray) object))
                        .toArray(IValue[]::new);

        NativeResource<Long> resultIValue;
        if (pastKeyValues == null) {
            resultIValue = blocks[0].forward(inputNative);
        } else {
            resultIValue =
                    blocks[1].forward(
                            new IValue[] {
                                inputNative[0],
                                inputNative[1],
                                inputNative[2],
                                (IValue) pastKeyValues
                            });
        }
        IValue[] resultIValueArray = ((IValue) resultIValue).toIValueTuple();

        manager.attachInternal("inputNative", inputNative);
        manager.attachInternal("resultIValueArray", resultIValueArray);
        manager.attachInternal("resultIValue", resultIValue);

        return new CausalLMOutput(
                resultIValueArray[0].toNDList(manager).singletonOrThrow(), resultIValueArray[1]);
    }

    public void poc(String inputType)
            throws ModelNotFoundException, MalformedModelException, IOException {
        NDManager manager = NDManager.newBaseManager();

        /////////////////////////////////////////////
        // Inference without cached key_values input
        /////////////////////////////////////////////

        // Load model for init inference
        String modelUrl =
                "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2_init.pt";
        Criteria<NDList, NDList> criteria_init =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(modelUrl)
                        .optProgress(new ProgressBar())
                        .optEngine("PyTorch")
                        .optOption("trainParam", String.valueOf(false))
                        .build();
        ZooModel<NDList, NDList> generator_init = criteria_init.loadModel();
        Block block_init = generator_init.getBlock();

        // Prepare input
        int[] inputArray = {40, 2883, 6155, 351, 616, 13779};

        NDArray inputIds = manager.create(inputArray);
        int numBatch = 1;
        if ("simple".equals(inputType)) {
            inputIds = manager.create(inputArray, new Shape(1, inputArray.length));
        } else if ("batch".equals(inputType)) {
            inputIds = manager.create(inputArray, new Shape(2, inputArray.length / 2));
            numBatch = 2;
        }

        NDArray positionIds =
                manager.arange(0, inputIds.getShape().size(-1), 1, DataType.INT64)
                        .reshape(1, -1)
                        .repeat(0, numBatch);

        NDArray attentionMask = manager.ones(positionIds.getShape());
        NDList input = new NDList(inputIds, positionIds, attentionMask);
        IValue[] inputNativeInit =
                input.stream()
                        .map(object -> IValue.from((PtNDArray) object))
                        .toArray(IValue[]::new);

        // inference
        NativeResource<Long> resultIValue = block_init.forward(inputNativeInit);
        IValue[] resultIValueArray = ((IValue) resultIValue).toIValueTuple();

        manager.attachInternal("inputNativeInit", inputNativeInit);
        manager.attachInternal("resultIValueArray", resultIValueArray);

        /////////////////////////////////////////////
        // Inference with cached key_values input
        /////////////////////////////////////////////

        // Load model
        modelUrl =
                "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2.pt";
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(modelUrl)
                        .optProgress(new ProgressBar())
                        .optEngine("PyTorch")
                        .optOption("trainParam", String.valueOf(false))
                        .build();

        ZooModel<NDList, NDList> generator = criteria.loadModel();
        Block block = generator.getBlock();

        // Prepare input
        long pastSeqLen = resultIValueArray[1].toNDList(manager).get(0).getShape().size(-2);
        if ("simple".equals(inputType)) {
            inputIds = manager.create(new int[] {404}, new Shape(1, 1));
        } else if ("batch".equals(inputType)) {
            inputIds = manager.create(new int[] {404, 403, 402, 401}, new Shape(numBatch, 2));
        }
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
        input = new NDList(inputIds, positionIds, attentionMask);

        IValue[] inputNative =
                input.stream()
                        .map(object -> IValue.from((PtNDArray) object))
                        .toArray(IValue[]::new);

        // Inference
        // Here resultIValueArray[1] = past_key_values, which is from the first inference
        // and has been used here as a demo cached input.
        NativeResource<Long> output2 =
                block.forward(
                        new IValue[] {
                            inputNative[0], inputNative[1], inputNative[2], resultIValueArray[1]
                        });
        NDList result = (output2).toNDList(manager);

        manager.attachInternal("inputNative", inputNative);

        ////////////////////////
        // close resources
        ////////////////////////
        manager.close();
        generator_init.close();
        generator.close();
    }

    @Override
    public void close() {
        models.forEach(ZooModel::close);
    }
}
