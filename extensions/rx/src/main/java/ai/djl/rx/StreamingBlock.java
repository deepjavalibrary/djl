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
package ai.djl.rx;

import ai.djl.ndarray.NDList;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import io.reactivex.rxjava3.core.Flowable;

/**
 * A {@link Block} that supports {@link #forwardStream(ParameterStore, NDList, boolean, PairList)}.
 */
public interface StreamingBlock extends Block {

    /**
     * Applies the operating function of the block once. This method should be called only on blocks
     * that are initialized.
     *
     * @param parameterStore the parameter store
     * @param inputs the input NDList
     * @param training true for a training forward pass (turn on dropout and layerNorm)
     * @return the output of the forward pass
     */
    default Flowable<NDList> forwardStream(
            ParameterStore parameterStore, NDList inputs, boolean training) {
        return forwardStream(parameterStore, inputs, training, null);
    }

    /**
     * Applies the operating function of the block once. This method should be called only on blocks
     * that are initialized.
     *
     * @param parameterStore the parameter store
     * @param inputs the input NDList
     * @param training true for a training forward pass (turn on dropout and layerNorm)
     * @param params optional parameters
     * @return the output of the forward pass
     */
    Flowable<NDList> forwardStream(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params);
}
