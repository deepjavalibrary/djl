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

public interface StepGenerator extends AutoCloseable {

    /**
     * @param input input
     * @param pastKeyValues past_key_values
     * @param manager manager
     * @return CausalLMOutput
     */
    CausalLMOutput stepGeneration(
            NDList input, NativeResource<Long> pastKeyValues, NDManager manager);

    void poc(String inputType) throws ModelNotFoundException, MalformedModelException, IOException;

    /** {@inheritDoc} */
    @Override
    void close();
}
