/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.util.passthrough;

import ai.djl.ndarray.NDList;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

/**
 * A translator that stores and removes data from a {@link PassthroughNDArray}.
 *
 * @param <I> translator input type
 * @param <O> translator output type
 */
public class PassthroughTranslator<I, O> implements NoBatchifyTranslator<I, O> {

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, I input) throws Exception {
        return new NDList(new PassthroughNDArray(input));
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public O processOutput(TranslatorContext ctx, NDList list) {
        PassthroughNDArray wrapper = (PassthroughNDArray) list.singletonOrThrow();
        return (O) wrapper.getObject();
    }
}
