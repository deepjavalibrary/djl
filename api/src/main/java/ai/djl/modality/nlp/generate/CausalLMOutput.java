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
package ai.djl.modality.nlp.generate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

/** CausalLMOuput is used to contain multiple output of a language model. */
public class CausalLMOutput {

    // [batch, seq, feature]
    // The prob. conditional on a sequence that ends at an element in seq-dim. seq-dim-size =
    // |inputIds|
    private NDArray logits;

    // [batch, seq, dim] * (layers+1) -> take -1
    // The vec. rep. of a sequence that ends at an element in seq-dim. seq-dim-size = |inputIds|
    private NDArray hiddenStates;

    // (k, v) * numLayer,
    // kv: [batch, heads, seq_past, feature]
    // The cache of past sequence. seq-dim-size == |seq_past| + |inputIds|
    private NDList pastKeyValuesList;

    /**
     * Constructs a new {@code CausalLMOutput} instance.
     *
     * @param logits the logits NDArray
     * @param pastKeyValues the key-value cache
     */
    public CausalLMOutput(NDArray logits, NDList pastKeyValues) {
        this.logits = logits;
        this.pastKeyValuesList = pastKeyValues;
    }

    /**
     * Constructs a new {@code CausalLMOutput} intance.
     *
     * @param logits the logits NDArray
     * @param hiddenState the first layer hiddenStates used as word embedding
     * @param pastKeyValueList the key-value cache
     */
    public CausalLMOutput(NDArray logits, NDArray hiddenState, NDList pastKeyValueList) {
        this.logits = logits;
        this.pastKeyValuesList = pastKeyValueList;
        this.hiddenStates = hiddenState;
    }

    /**
     * Returns the value of the logits.
     *
     * @return the value of logits
     */
    public NDArray getLogits() {
        return logits;
    }

    /**
     * Sets the value of the logits.
     *
     * @param logits value of logits NDArray
     */
    public void setLogits(NDArray logits) {
        this.logits = logits;
    }

    /**
     * Returns the value of the allHiddenStates.
     *
     * @return the value of allHiddenStates
     */
    public NDArray getHiddenState() {
        return hiddenStates;
    }

    /**
     * Returns the value of the pastKeyValuesList.
     *
     * @return the value of pastKeyValuesList
     */
    public NDList getPastKeyValuesList() {
        return pastKeyValuesList;
    }
}
