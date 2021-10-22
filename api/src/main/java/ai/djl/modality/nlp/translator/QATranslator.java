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
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import java.util.Locale;
import java.util.Map;

/** An abstract class to define the question answering translator. */
public abstract class QATranslator implements Translator<QAInput, String> {

    protected Batchifier batchifier;
    protected String tokenizerName;
    protected String vocab;
    protected Locale locale;
    protected boolean toLowerCase;
    protected boolean includeTokenTypes;
    protected boolean padding;
    protected boolean truncation;
    protected int maxLength;
    protected int maxLabels;

    protected QATranslator(BaseBuilder<?> builder) {
        this.batchifier = builder.batchifier;
        this.tokenizerName = builder.tokenizerName;
        this.vocab = builder.vocab;
        this.locale = builder.locale;
        this.toLowerCase = builder.toLowerCase;
        this.includeTokenTypes = builder.includeTokenTypes;
        this.padding = builder.padding;
        this.truncation = builder.truncation;
        this.maxLength = builder.maxLength;
        this.maxLabels = builder.maxLabels;
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
        String tokenizerName;
        String vocab = "vocab.txt";
        Locale locale = Locale.ROOT;
        boolean toLowerCase;
        boolean includeTokenTypes;
        boolean padding;
        boolean truncation;
        int maxLength = 128;
        int maxLabels;

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

        /**
         * Sets the name of the tokenizer for the {@link Translator}.
         *
         * @param tokenizer the name of the tokenizer
         * @return this builder
         */
        public T optTokenizer(String tokenizer) {
            this.tokenizerName = tokenizer;
            return self();
        }

        /**
         * Sets the name of the vocabulary file for the {@link Translator}.
         *
         * @param vocab name of the vocabulary file
         * @return this builder
         */
        public T optVocab(String vocab) {
            if (vocab != null) {
                this.vocab = vocab;
            }
            return self();
        }

        /**
         * Sets the name of the locale for the {@link Translator}.
         *
         * @param locale the name of the locale
         * @return this builder
         */
        public T optLocale(String locale) {
            if (locale != null) {
                this.locale = Locale.forLanguageTag(locale);
            }
            return self();
        }

        /**
         * Sets the if convert text to lower case for the {@link Translator}.
         *
         * @param toLowerCase if convert text to lower case
         * @return this builder
         */
        public T optToLowerCase(boolean toLowerCase) {
            this.toLowerCase = toLowerCase;
            return self();
        }

        /**
         * Sets the if include token types for the {@link Translator}.
         *
         * @param includeTokenTypes if include token types
         * @return this builder
         */
        public T optIncludeTokenTypes(boolean includeTokenTypes) {
            this.includeTokenTypes = includeTokenTypes;
            return self();
        }

        /**
         * Sets the if pad the tokens for the {@link Translator}.
         *
         * @param padding if pad the tokens
         * @return this builder
         */
        public T optPadding(boolean padding) {
            this.padding = padding;
            return self();
        }

        /**
         * Sets the if truncate the tokens for the {@link Translator}.
         *
         * @param truncation if truncate the tokens
         * @return this builder
         */
        public T optTruncation(boolean truncation) {
            this.truncation = truncation;
            return self();
        }

        /**
         * Sets the max number of tokens for the {@link Translator}.
         *
         * @param maxLength the max number of tokens
         * @return this builder
         */
        public T optMaxLength(int maxLength) {
            this.maxLength = maxLength;
            return self();
        }

        /**
         * Sets the max number of labels for the {@link Translator}.
         *
         * @param maxLabels the max number of labels
         * @return this builder
         */
        public T optMaxLabels(int maxLabels) {
            this.maxLabels = maxLabels;
            return self();
        }

        /**
         * Configures the builder with the model arguments.
         *
         * @param arguments the model arguments
         */
        public void configure(Map<String, ?> arguments) {
            optTokenizer(ArgumentsUtil.stringValue(arguments, "tokenizer"));
            optVocab(ArgumentsUtil.stringValue(arguments, "vocab"));
            optLocale(ArgumentsUtil.stringValue(arguments, "locale"));
            optToLowerCase(ArgumentsUtil.booleanValue(arguments, "toLowerCase"));
            optIncludeTokenTypes(ArgumentsUtil.booleanValue(arguments, "includeTokenTypes"));
            optPadding(ArgumentsUtil.booleanValue(arguments, "padding"));
            optTruncation(ArgumentsUtil.booleanValue(arguments, "truncation"));
            optMaxLength(ArgumentsUtil.intValue(arguments, "maxLength", 128));
            optMaxLabels(ArgumentsUtil.intValue(arguments, "maxLabels"));
        }

        protected abstract T self();
    }
}
