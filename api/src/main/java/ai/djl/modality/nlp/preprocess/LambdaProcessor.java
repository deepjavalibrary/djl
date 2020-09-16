/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.nlp.preprocess;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * {@code TextProcessor} will apply user defined lambda function on input tokens.
 *
 * <p>The function can only support single input and output.
 */
public class LambdaProcessor implements TextProcessor {
    private Function<String, String> processor;

    /**
     * Creates a {@code LambdaProcessor} and specify the function to apply.
     *
     * @param processor The lambda function to apply on input String
     */
    public LambdaProcessor(Function<String, String> processor) {
        this.processor = processor;
    }

    /** {@inheritDoc} */
    @Override
    public List<String> preprocess(List<String> tokens) {
        return tokens.stream().map(processor).collect(Collectors.toList());
    }
}
