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

package ai.djl.modality.nlp.translator;

import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;

/** An abstract class to define the question answering translator. */
public abstract class QATranslator implements Translator<QAInput, String> {

    protected Batchifier batchifier;

    protected QATranslator(BaseBuilder<?> builder) {
        this.batchifier = builder.batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /** The builder for question answering translator. */
    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        Batchifier batchifier = Batchifier.STACK;

        /**
         * Sets the {@link Batchifier} for the {@link Translator}.
         *
         * @param batchifier the {@link Batchifier} to be set
         * @return this builder
         */
        public T optBatchifier(Batchifier batchifier) {
            this.batchifier = batchifier;
            return self();
        }

        protected abstract T self();
    }
}
