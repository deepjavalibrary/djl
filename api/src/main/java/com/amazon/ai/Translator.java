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
package com.amazon.ai;

import com.amazon.ai.ndarray.NDList;

/**
 * The <code>Translator</code> interface provides model preprocessing and postprocessing
 * functionality.
 *
 * <p>Users can use this in {@link com.amazon.ai.inference.Predictor} with Input and Output Object
 * Specified
 */
public interface Translator<I, O> {

    /**
     * Process the input and convert to NDList
     *
     * @param ctx Toolkit that would help to creating input NDArray
     * @param input Input Object
     * @return {@link NDList}
     */
    NDList processInput(TranslatorContext ctx, I input);

    /**
     * Process output NDList to the corresponding Output Object
     *
     * @param ctx Toolkit used to do postprocessing
     * @param list Output NDList after inference
     * @return Ouput Object
     */
    O processOutput(TranslatorContext ctx, NDList list);
}
