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
package ai.djl.translate;

import ai.djl.ndarray.NDList;

/**
 * A {@link Translator} made by combining a {@link PreProcessor} and a {@link PostProcessor}.
 *
 * @param <I> the input class
 * @param <O> the output class
 */
public class BasicTranslator<I, O> implements Translator<I, O> {

    private PreProcessor<I> preProcessor;
    private PostProcessor<O> postProcessor;
    private Batchifier batchifier;

    /**
     * Constructs a {@link BasicTranslator} with the default {@link Batchifier}.
     *
     * @param preProcessor the preProcessor to use for pre-processing
     * @param postProcessor the postProcessor to use for post-processing
     */
    public BasicTranslator(PreProcessor<I> preProcessor, PostProcessor<O> postProcessor) {
        this.preProcessor = preProcessor;
        this.postProcessor = postProcessor;
    }

    /**
     * Constructs a {@link BasicTranslator}.
     *
     * @param preProcessor the preProcessor to use for pre-processing
     * @param postProcessor the postProcessor to use for post-processing
     * @param batchifier the batchifier to use
     */
    public BasicTranslator(
            PreProcessor<I> preProcessor, PostProcessor<O> postProcessor, Batchifier batchifier) {
        this.preProcessor = preProcessor;
        this.postProcessor = postProcessor;
        this.batchifier = batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public O processOutput(TranslatorContext ctx, NDList list) throws Exception {
        return postProcessor.processOutput(ctx, list);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, I input) throws Exception {
        return preProcessor.processInput(ctx, input);
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        if (batchifier != null) {
            return batchifier;
        }
        return Translator.super.getBatchifier();
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        if (preProcessor instanceof Translator) {
            ((Translator<?, ?>) preProcessor).prepare(ctx);
        }
        if (postProcessor instanceof Translator && postProcessor != preProcessor) {
            ((Translator<?, ?>) postProcessor).prepare(ctx);
        }
    }
}
