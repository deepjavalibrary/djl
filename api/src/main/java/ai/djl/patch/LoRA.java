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
package ai.djl.patch;

import ai.djl.ndarray.NDArray;
import ai.djl.util.Pair;

import java.util.Map;

/**
 * A {@link ParamPatch} based on low-rank adapters.
 *
 * <p>Based on the paper <a href="https://arxiv.org/abs/2106.09685">LoRA: Low-Rank Adaptation of
 * Large Language Models</a>.
 *
 * <p>TODO This support for LoRA is still a placeholder and needs effective code for creating and
 * training
 */
public class LoRA extends ParamPatch {

    /** Data of type map from param name to (A, B) pair. */
    Map<String, Pair<NDArray, NDArray>> data;

    /**
     * Constructs a {@link LoRA}.
     *
     * @param data the data to patch with
     */
    public LoRA(Map<String, Pair<NDArray, NDArray>> data) {
        this.data = data;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getPatch(String paramName) {
        Pair<NDArray, NDArray> d = data.get(paramName);
        return d.getKey().get(paramName).matMul(d.getValue().get(paramName));
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        for (Pair<NDArray, NDArray> d : data.values()) {
            d.getKey().close();
            d.getValue().close();
        }
    }
}
