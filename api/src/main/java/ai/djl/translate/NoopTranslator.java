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
package ai.djl.translate;

import ai.djl.ndarray.NDList;

/** A no operational {@link Translator} implementation. */
public class NoopTranslator implements Translator<NDList, NDList> {

    private Batchifier batchifier;

    /**
     * Constructs a {@link NoopTranslator} with the given {@link Batchifier}.
     *
     * @param batchifier batchifier to use
     */
    public NoopTranslator(Batchifier batchifier) {
        this.batchifier = batchifier;
    }

    /** Constructs a {@link NoopTranslator}. */
    public NoopTranslator() {}

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, NDList input) {
        return input;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processOutput(TranslatorContext ctx, NDList list) {
        return list;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /**
     * Sets the {@link Batchifier} for the Translator.
     *
     * @param batchifier the {@link Batchifier} for the Translator
     */
    public void setBatchifier(Batchifier batchifier) {
        this.batchifier = batchifier;
    }
}
