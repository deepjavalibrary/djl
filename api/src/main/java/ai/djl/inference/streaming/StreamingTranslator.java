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
package ai.djl.inference.streaming;

import ai.djl.ndarray.NDList;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.stream.Stream;

/**
 * An expansion of the {@link Translator} with postProcessing for the {@link StreamingBlock} (used
 * by {@link ai.djl.inference.Predictor#streamingPredict(Object)}.
 *
 * @param <I> the input type
 * @param <O> the output type
 */
public interface StreamingTranslator<I, O> extends Translator<I, O> {

    /**
     * Processes the output NDList to the corresponding output object.
     *
     * @param ctx the toolkit used for post-processing
     * @param list the output NDList after inference, usually immutable in engines like
     *     PyTorch. @see <a href="https://github.com/deepjavalibrary/djl/issues/1774">Issue 1774</a>
     * @return the output object of expected type
     * @throws Exception if an error occurs during processing output
     */
    @SuppressWarnings("PMD.SignatureDeclareThrowsException")
    O processStreamOutput(TranslatorContext ctx, Stream<NDList> list) throws Exception;
}
