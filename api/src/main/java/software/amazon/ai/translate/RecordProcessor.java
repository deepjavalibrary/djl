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
package software.amazon.ai.translate;

import software.amazon.ai.training.dataset.Record;
import software.amazon.ai.util.Pair;

public interface RecordProcessor<I, L> {

    /**
     * Processes the input and label pair and converts it to {@link Record}.
     *
     * @param ctx Toolkit that would help to creating input NDArray
     * @param input Input Object
     * @param label Label Object
     * @return {@link Record} containing the converted input and converted label
     * @throws Exception if an error occurs during processing input
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    Record processInput(TranslatorContext ctx, I input, L label) throws Exception;

    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    default Record processInput(TranslatorContext ctx, Pair<I, L> pair) throws Exception {
        return processInput(ctx, pair.getKey(), pair.getValue());
    }
}
