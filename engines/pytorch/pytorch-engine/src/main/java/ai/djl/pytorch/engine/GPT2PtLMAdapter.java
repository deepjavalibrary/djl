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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.pytorch.jni.IValue;
import ai.djl.pytorch.jni.IValueUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.CausalLMOutput;
import ai.djl.translate.GPTConfig;
import ai.djl.translate.LMAdapter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class GPT2PtLMAdapter implements LMAdapter {
    Block[] blocks;
    List<ZooModel<NDList, NDList>> models;
    GPTConfig config;

    public GPT2PtLMAdapter(String[] modelUrls)
            throws ModelNotFoundException, MalformedModelException, IOException {
        config = new GPTConfig();

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
    public CausalLMOutput forward(NDList input, NDList pastKeyValues, NDManager manager) {
        IValue[] inputNative =
                input.stream()
                        .map(object -> IValue.from((PtNDArray) object))
                        .toArray(IValue[]::new);

        IValue resultIValue;
        if (pastKeyValues == null) {
            resultIValue = (IValue) blocks[0].forward(inputNative);
        } else {
            resultIValue =
                    (IValue)
                            blocks[1].forward(
                                    new IValue[] {
                                        inputNative[0],
                                        inputNative[1],
                                        inputNative[2],
                                        IValueUtils.toTupleIValue(
                                                pastKeyValues, new long[] {config.numLayers, 2})
                                    });
        }
        NDList output = resultIValue.toNDList(manager);

        manager.attachInternal("inputNative", inputNative);
        manager.attachInternal("resultIValue", resultIValue);

        return new CausalLMOutput(output.get(0), output.subNDList(1));
    }

    @Override
    public void close() {
        models.forEach(ZooModel::close);
    }
}
