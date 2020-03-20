/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

/**
 * {@code TextProcessor} allows applying pre-processing to input tokens for natural language
 * applications. Multiple implementations of {@code TextProcessor} can be applied on the same input.
 * The order of application of different implementations of {@code TextProcessor} can make a
 * difference in the final output.
 */
public interface TextProcessor {

    /**
     * Applies the preprocessing defined to the given input tokens.
     *
     * @param tokens the tokens created after the input text is tokenized
     * @return the preprocessed tokens
     */
    List<String> preprocess(List<String> tokens);
}
