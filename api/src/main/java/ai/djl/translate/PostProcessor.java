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
 * An interface that provides post-processing functionality.
 *
 * @param <O> the type of the output object expected
 */
public interface PostProcessor<O> {

    /**
     * Processes the output NDList to the corresponding output object.
     *
     * @param ctx the toolkit used for post-processing
     * @param list the output NDList after inference
     * @return the output object of expected type
     * @throws Exception if an error occurs during processing output
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    O processOutput(TranslatorContext ctx, NDList list) throws Exception;
}
