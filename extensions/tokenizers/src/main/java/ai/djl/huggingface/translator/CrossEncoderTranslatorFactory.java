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
package ai.djl.huggingface.translator;

import ai.djl.Model;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.nlp.translator.CrossEncoderServingTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import ai.djl.util.StringPair;

import java.io.IOException;
import java.io.Serializable;
import java.lang.reflect.Type;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** A {@link TranslatorFactory} that creates a {@link CrossEncoderTranslatorFactory} instance. */
public class CrossEncoderTranslatorFactory implements TranslatorFactory, Serializable {

    private static final long serialVersionUID = 1L;

    private static final Set<Pair<Type, Type>> SUPPORTED_TYPES = new HashSet<>();

    static {
        SUPPORTED_TYPES.add(new Pair<>(StringPair.class, float[].class));
        SUPPORTED_TYPES.add(new Pair<>(Input.class, Output.class));
    }

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return SUPPORTED_TYPES;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Model model, Map<String, ?> arguments)
            throws TranslateException {
        Path modelPath = model.getModelPath();
        try {
            HuggingFaceTokenizer tokenizer =
                    HuggingFaceTokenizer.builder(arguments)
                            .optTokenizerPath(modelPath)
                            .optManager(model.getNDManager())
                            .build();
            CrossEncoderTranslator translator =
                    CrossEncoderTranslator.builder(tokenizer, arguments).build();
            if (input == StringPair.class && output == float[].class) {
                return (Translator<I, O>) translator;
            } else if (input == Input.class && output == Output.class) {
                return (Translator<I, O>) new CrossEncoderServingTranslator(translator);
            }
            throw new IllegalArgumentException("Unsupported input/output types.");
        } catch (IOException e) {
            throw new TranslateException("Failed to load tokenizer.", e);
        }
    }
}
