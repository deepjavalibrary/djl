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
package ai.djl.translate;

import ai.djl.ndarray.NDList;

/**
 * An interface that provides pre-processing functionality.
 *
 * @param <I> the type of the input object
 */
public interface PreProcessor<I> {

    /**
     * Processes the input and converts it to NDList.
     *
     * @param ctx the toolkit for creating the input NDArray
     * @param input the input object
     * @return the {@link NDList} after pre-processing
     * @throws Exception if an error occurs during processing input
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    NDList processInput(TranslatorContext ctx, I input) throws Exception;
}
