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
package ai.djl.translate;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.util.NativeResource;

import java.io.IOException;

/**
 * This is a wrapper over the model files from different sources, e.g. gpt2.pt, gpt2.onnx, etc. This
 * interface is an abstraction of the causal language model, which in essence is a conditional
 * probability function: p_\theta(v_t | x_{< t})}, v_t \in V. i.e. given the past tokens up to a
 * certain time x_{< t}, the probability that the next token is v, taken from a vocabulary set V.
 * \theta is the model's weight. This function can take an input sequence `inputIds`, whose length
 * can be greater than one. In this case, the output is still p_\theta(v_i | x_{< i})}, i in
 * range(|inputIds|). This means for each i, the output probability is conditional on the past
 * sequence up to i.
 */
public interface StepGenerator extends AutoCloseable {

    /**
     * @param input input
     * @param pastKeyValues past_key_values
     * @param manager manager
     * @return CausalLMOutput
     */

    // Will be deprecated
    default CausalLMOutput stepGeneration(
            NDList input, NativeResource<Long> pastKeyValues, NDManager manager) {
        return null;
    }

    default CausalLMOutput stepGeneration2(NDList ndList, NDList pastKeyValues, NDManager manager) {
        return null;
    }

    void poc(String inputType) throws ModelNotFoundException, MalformedModelException, IOException;

    /** {@inheritDoc} */
    @Override
    void close();
}
